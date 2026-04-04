// Circle FRI Engine — GPU-accelerated Fast Reed-Solomon IOP over Mersenne31
// Core commitment/query protocol for Circle STARKs (Stwo-style).
//
// Unlike standard FRI over multiplicative subgroups, Circle FRI works on the
// circle group x^2 + y^2 = 1 mod p where p = 2^31 - 1:
//   - First fold uses y-coordinate (twin-coset decomposition): pairs (x,y) and (x,-y)
//   - Subsequent folds use x-coordinate squaring map: x -> 2x^2 - 1
//   - Domain halving: circle of order 2^k -> x-coords of order 2^(k-1) -> 2^(k-2) via squaring

import Foundation
import Metal

public class CircleFRIEngine {
    public static let version = Versions.circleFRI

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let foldFirstFunction: MTLComputePipelineState   // y-coordinate fold
    let foldFunction: MTLComputePipelineState         // x-coordinate fold
    let foldFused2Function: MTLComputePipelineState   // fused y+x fold

    // Reuse Circle NTT engine for LDE if needed
    public let circleNTT: CircleNTTEngine

    // Cached twiddle buffers: inv_2y and inv_2x per logN
    private var inv2yCache: [Int: MTLBuffer] = [:]
    private var inv2xCache: [Int: [MTLBuffer]] = [:]  // keyed by logN, array indexed by fold round

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

        let library = try CircleFRIEngine.compileShaders(device: device)

        guard let foldFirstFn = library.makeFunction(name: "circle_fri_fold_first"),
              let foldFn = library.makeFunction(name: "circle_fri_fold"),
              let foldFused2Fn = library.makeFunction(name: "circle_fri_fold_fused2") else {
            throw MSMError.missingKernel
        }

        self.foldFirstFunction = try device.makeComputePipelineState(function: foldFirstFn)
        self.foldFunction = try device.makeComputePipelineState(function: foldFn)
        self.foldFused2Function = try device.makeComputePipelineState(function: foldFused2Fn)

        self.circleNTT = try CircleNTTEngine()
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let fieldSource = try String(contentsOfFile: shaderDir + "/fields/mersenne31.metal", encoding: .utf8)
        let friSource = try String(contentsOfFile: shaderDir + "/fri/circle_fri.metal", encoding: .utf8)

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

    /// Precompute 1/(2*y_i) for the first fold (y-coordinate twin-coset decomposition).
    /// Domain points are on the circle coset of size 2^logN.
    private func getInv2y(logN: Int) -> MTLBuffer {
        if let cached = inv2yCache[logN] { return cached }

        let n = 1 << logN
        let half = n / 2
        let domain = circleCosetDomain(logN: logN)

        var inv2y = [M31](repeating: M31.zero, count: half)
        let two = M31(v: 2)
        for i in 0..<half {
            let twoY = m31Mul(two, domain[i].y)
            inv2y[i] = m31Inverse(twoY)
        }

        let buf = createM31Buffer(inv2y)!
        inv2yCache[logN] = buf
        return buf
    }

    /// Precompute 1/(2*x_i) for subsequent folds (x-coordinate squaring map).
    /// After the first fold, the domain is projected to x-coordinates.
    /// After each subsequent fold, the squaring map x -> 2x^2 - 1 halves the domain.
    /// Returns array of buffers, one per x-fold round.
    private func getInv2x(logN: Int) -> [MTLBuffer] {
        if let cached = inv2xCache[logN] { return cached }

        let n = 1 << logN
        let half = n / 2
        let domain = circleCosetDomain(logN: logN)

        // After y-fold: domain becomes x-coordinates of the first half
        var xs = (0..<half).map { domain[$0].x }

        var bufs: [MTLBuffer] = []
        let two = M31(v: 2)

        // Each x-fold halves the domain size
        var currentSize = half
        while currentSize > 1 {
            let foldHalf = currentSize / 2
            var inv2x = [M31](repeating: M31.zero, count: foldHalf)
            for i in 0..<foldHalf {
                let twoX = m31Mul(two, xs[i])
                inv2x[i] = m31Inverse(twoX)
            }
            bufs.append(createM31Buffer(inv2x)!)

            // Apply squaring map: x -> 2x^2 - 1 for next round
            var newXs = [M31](repeating: M31.zero, count: foldHalf)
            for i in 0..<foldHalf {
                newXs[i] = m31Sub(m31Mul(two, m31Sqr(xs[i])), M31.one)
            }
            xs = newXs
            currentSize = foldHalf
        }

        inv2xCache[logN] = bufs
        return bufs
    }

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

    // MARK: - Single Fold (GPU)

    /// Perform one Circle FRI fold step on GPU.
    /// If isFirstFold is true, uses y-coordinate twiddles; otherwise uses x-coordinate twiddles.
    public func fold(evals: [M31], alpha: M31, logN: Int, isFirstFold: Bool,
                     xFoldRound: Int = 0) throws -> [M31] {
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
        var alphaBuf = alpha
        var nVal = UInt32(n)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!

        if isFirstFold {
            let inv2yBuf = getInv2y(logN: logN)
            enc.setComputePipelineState(foldFirstFunction)
            enc.setBuffer(evalsBuf, offset: 0, index: 0)
            enc.setBuffer(outputBuf, offset: 0, index: 1)
            enc.setBuffer(inv2yBuf, offset: 0, index: 2)
            enc.setBytes(&alphaBuf, length: stride, index: 3)
            enc.setBytes(&nVal, length: 4, index: 4)
        } else {
            let inv2xBufs = getInv2x(logN: logN)
            precondition(xFoldRound < inv2xBufs.count)
            enc.setComputePipelineState(foldFunction)
            enc.setBuffer(evalsBuf, offset: 0, index: 0)
            enc.setBuffer(outputBuf, offset: 0, index: 1)
            enc.setBuffer(inv2xBufs[xFoldRound], offset: 0, index: 2)
            enc.setBytes(&alphaBuf, length: stride, index: 3)
            enc.setBytes(&nVal, length: 4, index: 4)
        }

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
    /// First challenge uses y-fold, rest use x-fold.
    /// Single command buffer with memory barriers between rounds.
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

        // Precompute all twiddles
        let inv2yBuf = getInv2y(logN: logN)
        let inv2xBufs = getInv2x(logN: logN)

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

            if i == 0 {
                // First fold: y-coordinate
                enc.setComputePipelineState(foldFirstFunction)
                enc.setBuffer(currentBuf, offset: 0, index: 0)
                enc.setBuffer(outputBuf, offset: 0, index: 1)
                enc.setBuffer(inv2yBuf, offset: 0, index: 2)
                enc.setBytes(&alpha, length: stride, index: 3)
                enc.setBytes(&nVal, length: 4, index: 4)
            } else {
                // Subsequent folds: x-coordinate
                enc.setComputePipelineState(foldFunction)
                enc.setBuffer(currentBuf, offset: 0, index: 0)
                enc.setBuffer(outputBuf, offset: 0, index: 1)
                enc.setBuffer(inv2xBufs[i - 1], offset: 0, index: 2)
                enc.setBytes(&alpha, length: stride, index: 3)
                enc.setBytes(&nVal, length: 4, index: 4)
            }

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
    /// Uses CPU Merkle (Keccak-style hash of M31 bytes) since M31 Poseidon2 is not yet available.
    public func commitPhase(evals: [M31], alphas: [M31]) throws -> CircleFRICommitment {
        let n = evals.count
        precondition(n > 1 && (n & (n - 1)) == 0)
        let logN = Int(log2(Double(n)))
        precondition(alphas.count <= logN)

        let stride = MemoryLayout<M31>.stride

        // Precompute twiddles
        let inv2yBuf = getInv2y(logN: logN)
        let inv2xBufs = getInv2x(logN: logN)

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

            if i == 0 {
                enc.setComputePipelineState(foldFirstFunction)
                enc.setBuffer(layerBufs[0], offset: 0, index: 0)
                enc.setBuffer(layerBufs[1], offset: 0, index: 1)
                enc.setBuffer(inv2yBuf, offset: 0, index: 2)
                enc.setBytes(&alpha, length: stride, index: 3)
                enc.setBytes(&nVal, length: 4, index: 4)
            } else {
                enc.setComputePipelineState(foldFunction)
                enc.setBuffer(layerBufs[i], offset: 0, index: 0)
                enc.setBuffer(layerBufs[i + 1], offset: 0, index: 1)
                enc.setBuffer(inv2xBufs[i - 1], offset: 0, index: 2)
                enc.setBytes(&alpha, length: stride, index: 3)
                enc.setBytes(&nVal, length: 4, index: 4)
            }

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

            // CPU Merkle root: hash M31 values by treating as UInt32 bytes
            let root = cpuM31MerkleRoot(layer)
            roots.append(root)
        }
        let merkleTime = (CFAbsoluteTimeGetCurrent() - merkleT0) * 1000

        if profileCommit {
            fputs(String(format: "  circleFRI commitPhase: fold %.1fms, merkle %.1fms, total %.1fms\n",
                        foldTime, merkleTime, foldTime + merkleTime), stderr)
        }

        let finalLayer = layers.last!
        let finalValue = finalLayer.count == 1 ? finalLayer[0] : finalLayer[0]

        return CircleFRICommitment(
            layers: layers,
            roots: roots,
            alphas: alphas,
            finalValue: finalValue,
            logN: logN
        )
    }

    // MARK: - Query Phase

    /// Generate query proofs: for each query index, extract evaluation pairs
    /// and Merkle paths at each layer.
    public func queryPhase(commitment: CircleFRICommitment, queryIndices: [UInt32]) -> [CircleFRIQueryProof] {
        var proofs = [CircleFRIQueryProof]()
        proofs.reserveCapacity(queryIndices.count)

        for qi in 0..<queryIndices.count {
            var layerEvals: [(M31, M31)] = []
            var merklePaths: [[M31]] = []
            var idx = queryIndices[qi]

            for layer in 0..<(commitment.layers.count - 1) {
                let evals = commitment.layers[layer]
                let n = evals.count
                let halfN = UInt32(n / 2)

                // Paired elements: idx and idx + halfN (or idx - halfN)
                let lowerIdx = idx < halfN ? idx : idx - halfN
                let upperIdx = lowerIdx + halfN
                let evalA = evals[Int(lowerIdx)]
                let evalB = evals[Int(upperIdx)]
                layerEvals.append((evalA, evalB))

                // Merkle path for this layer
                let path = cpuM31MerklePath(evals, index: Int(idx))
                merklePaths.append(path)

                // Next layer index: fold maps to lower half
                idx = lowerIdx
            }

            proofs.append(CircleFRIQueryProof(
                initialIndex: queryIndices[qi],
                layerEvals: layerEvals,
                merklePaths: merklePaths
            ))
        }

        return proofs
    }

    // MARK: - Verification

    /// Verify a Circle FRI proof: check fold consistency at query positions.
    public func verify(commitment: CircleFRICommitment, queries: [CircleFRIQueryProof]) -> Bool {
        let logN = commitment.logN
        let domain = circleCosetDomain(logN: logN)
        let n = 1 << logN
        let two = M31(v: 2)

        // Precompute x-coordinates at each fold level for verification
        var xCoords: [[M31]] = []
        var xs = (0..<(n / 2)).map { domain[$0].x }
        var currentSize = n / 2
        while currentSize > 1 {
            xCoords.append(xs)
            let foldHalf = currentSize / 2
            var newXs = [M31](repeating: M31.zero, count: foldHalf)
            for i in 0..<foldHalf {
                newXs[i] = m31Sub(m31Mul(two, m31Sqr(xs[i])), M31.one)
            }
            xs = newXs
            currentSize = foldHalf
        }

        for query in queries {
            var idx = query.initialIndex

            for layer in 0..<(commitment.layers.count - 1) {
                let (evalA, evalB) = query.layerEvals[layer]
                let layerN = commitment.layers[layer].count
                let halfN = UInt32(layerN / 2)
                let alpha = commitment.alphas[layer]
                let lowerIdx = idx < halfN ? idx : idx - halfN

                // Compute expected folded value
                let sum = m31Add(evalA, evalB)
                let diff = m31Sub(evalA, evalB)
                let inv2 = M31(v: 1073741824)  // (p+1)/2

                var expected: M31
                if layer == 0 {
                    // y-fold: twiddle = 1/(2*y)
                    let y = domain[Int(lowerIdx)].y
                    let inv2y = m31Inverse(m31Mul(two, y))
                    let halfSum = m31Mul(sum, inv2)
                    let diffTerm = m31Mul(m31Mul(alpha, diff), inv2y)
                    expected = m31Add(halfSum, diffTerm)
                } else {
                    // x-fold: twiddle = 1/(2*x)
                    let xIdx = Int(lowerIdx)
                    precondition(layer - 1 < xCoords.count && xIdx < xCoords[layer - 1].count)
                    let x = xCoords[layer - 1][xIdx]
                    let inv2x = m31Inverse(m31Mul(two, x))
                    let halfSum = m31Mul(sum, inv2)
                    let diffTerm = m31Mul(m31Mul(alpha, diff), inv2x)
                    expected = m31Add(halfSum, diffTerm)
                }

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

    // MARK: - CPU Reference

    /// CPU-side Circle FRI fold for correctness verification.
    /// First fold uses y-coordinates, subsequent use x-coordinates.
    public static func cpuFold(evals: [M31], alpha: M31, logN: Int,
                                isFirstFold: Bool, domain: [CirclePoint]? = nil,
                                xCoords: [M31]? = nil) -> [M31] {
        let n = evals.count
        let half = n / 2
        let inv2 = M31(v: 1073741824)  // (p+1)/2
        let two = M31(v: 2)
        var folded = [M31](repeating: M31.zero, count: half)

        if isFirstFold {
            // y-fold: pair at (x,y) and (x,-y)
            let dom = domain ?? circleCosetDomain(logN: logN)
            for i in 0..<half {
                let a = evals[i]
                let b = evals[i + half]
                let halfSum = m31Mul(m31Add(a, b), inv2)
                let diff = m31Sub(a, b)
                let inv2y = m31Inverse(m31Mul(two, dom[i].y))
                let diffTerm = m31Mul(m31Mul(alpha, diff), inv2y)
                folded[i] = m31Add(halfSum, diffTerm)
            }
        } else {
            // x-fold: pair elements at i and i+half
            guard let xs = xCoords else {
                preconditionFailure("xCoords required for x-fold")
            }
            for i in 0..<half {
                let a = evals[i]
                let b = evals[i + half]
                let halfSum = m31Mul(m31Add(a, b), inv2)
                let diff = m31Sub(a, b)
                let inv2x = m31Inverse(m31Mul(two, xs[i]))
                let diffTerm = m31Mul(m31Mul(alpha, diff), inv2x)
                folded[i] = m31Add(halfSum, diffTerm)
            }
        }

        return folded
    }

    /// CPU multi-round fold for correctness testing.
    public static func cpuMultiFold(evals: [M31], alphas: [M31], logN: Int) -> [M31] {
        let n = 1 << logN
        let domain = circleCosetDomain(logN: logN)
        let two = M31(v: 2)

        var current = evals
        var xs = (0..<(n / 2)).map { domain[$0].x }

        for i in 0..<alphas.count {
            if i == 0 {
                current = cpuFold(evals: current, alpha: alphas[i], logN: logN,
                                   isFirstFold: true, domain: domain)
            } else {
                current = cpuFold(evals: current, alpha: alphas[i], logN: logN - i,
                                   isFirstFold: false, xCoords: xs)
                // Apply squaring map for next round
                let foldHalf = xs.count / 2
                var newXs = [M31](repeating: M31.zero, count: foldHalf)
                for j in 0..<foldHalf {
                    newXs[j] = m31Sub(m31Mul(two, m31Sqr(xs[j])), M31.one)
                }
                xs = newXs
            }
        }

        return current
    }

    // MARK: - CPU Merkle Helpers

    /// Simple CPU Merkle root over M31 array using a hash based on M31 values.
    /// Uses a simple hash: combine by (a*PRIME + b) mod p, iterated.
    /// This is a placeholder; plug in M31 Poseidon2 later.
    private func cpuM31MerkleRoot(_ leaves: [M31]) -> M31 {
        if leaves.count == 1 { return leaves[0] }
        var level = leaves
        while level.count > 1 {
            var next = [M31]()
            next.reserveCapacity(level.count / 2)
            for i in Swift.stride(from: 0, to: level.count, by: 2) {
                if i + 1 < level.count {
                    next.append(m31SimpleHash(level[i], level[i + 1]))
                } else {
                    next.append(level[i])
                }
            }
            level = next
        }
        return level[0]
    }

    /// Simple Merkle path extraction.
    private func cpuM31MerklePath(_ leaves: [M31], index: Int) -> [M31] {
        let n = leaves.count
        if n <= 1 { return [] }
        // Build full tree bottom-up
        var tree = [M31](repeating: M31.zero, count: 2 * n)
        for i in 0..<n { tree[n + i] = leaves[i] }
        var i = n - 1
        while i >= 1 {
            tree[i] = m31SimpleHash(tree[2 * i], tree[2 * i + 1])
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

/// Hash two M31 values into one. Uses a simple algebraic hash: (a * PRIME + b) mod p
/// combined with squaring for mixing. NOT cryptographically secure -- placeholder only.
@inline(__always)
func m31SimpleHash(_ a: M31, _ b: M31) -> M31 {
    // Mix: ((a * 1000000007) + b) * (a + b + 1) mod p
    let prime = M31(v: 1000000007 % M31.P)
    let t1 = m31Add(m31Mul(a, prime), b)
    let t2 = m31Add(m31Add(a, b), M31.one)
    return m31Mul(t1, t2)
}

// MARK: - Data Structures

public struct CircleFRICommitment {
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

public struct CircleFRIQueryProof {
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
