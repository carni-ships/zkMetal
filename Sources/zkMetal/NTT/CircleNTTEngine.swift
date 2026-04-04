// Circle NTT Engine — GPU-accelerated Circle FFT over Mersenne31
// The circle group over M31 has order p+1 = 2^31, giving full 2-adicity.
// Layer 0 uses y-coordinate twiddles (twin-coset decomposition).
// Layers 1..k-1 use x-coordinate twiddles with the squaring map.
// M31 elements are 4 bytes — same density as BabyBear.

import Foundation
import Metal

public class CircleNTTEngine {
    public static let version = Versions.circleNTT
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let butterflyFunction: MTLComputePipelineState
    let invButterflyFunction: MTLComputePipelineState
    let scaleFunction: MTLComputePipelineState

    private var fwdTwiddleCache: [Int: MTLBuffer] = [:]
    private var invTwiddleCache: [Int: MTLBuffer] = [:]
    private var invNCache: [Int: MTLBuffer] = [:]
    private var cachedDataBuf: MTLBuffer?
    private var cachedDataBufElements: Int = 0

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try CircleNTTEngine.compileShaders(device: device)

        guard let butterflyFn = library.makeFunction(name: "circle_ntt_butterfly"),
              let invButterflyFn = library.makeFunction(name: "circle_intt_butterfly"),
              let scaleFn = library.makeFunction(name: "circle_ntt_scale") else {
            throw MSMError.missingKernel
        }

        self.butterflyFunction = try device.makeComputePipelineState(function: butterflyFn)
        self.invButterflyFunction = try device.makeComputePipelineState(function: invButterflyFn)
        self.scaleFunction = try device.makeComputePipelineState(function: scaleFn)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let fieldSource = try String(contentsOfFile: shaderDir + "/fields/mersenne31.metal", encoding: .utf8)
        let nttSource = try String(contentsOfFile: shaderDir + "/ntt/ntt_circle.metal", encoding: .utf8)

        let cleanNTT = nttSource.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        let cleanField = fieldSource
            .replacingOccurrences(of: "#ifndef MERSENNE31_METAL", with: "")
            .replacingOccurrences(of: "#define MERSENNE31_METAL", with: "")
            .replacingOccurrences(of: "#endif // MERSENNE31_METAL", with: "")

        let combined = cleanField + "\n" + cleanNTT
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

    // MARK: - Twiddle computation

    /// Precompute forward twiddles for Circle NTT.
    /// For the forward NTT (DIT), layer 0 uses y-twiddles, layers 1+ use x-twiddles.
    /// We store them concatenated: [y-twiddles (N/2)] [x-level-1 (N/2)] [x-level-2 (N/4)] ...
    /// But for the GPU butterfly kernel, we need them indexed by (stage, position).
    /// For simplicity, we store one flat array of N/2 twiddles per stage.
    /// Stage 0: y-twiddles, Stage s (s>=1): x-twiddles at projection level s-1.
    private func getForwardTwiddles(logN: Int) -> MTLBuffer {
        if let cached = fwdTwiddleCache[logN] { return cached }
        let twiddles = circlePrecomputeForwardTwiddles(logN: logN)
        let buf = createM31Buffer(twiddles)!
        fwdTwiddleCache[logN] = buf
        return buf
    }

    private func getInverseTwiddles(logN: Int) -> MTLBuffer {
        if let cached = invTwiddleCache[logN] { return cached }
        let twiddles = circlePrecomputeInverseTwiddles(logN: logN)
        let buf = createM31Buffer(twiddles)!
        invTwiddleCache[logN] = buf
        return buf
    }

    private func getInvN(logN: Int) -> MTLBuffer {
        if let cached = invNCache[logN] { return cached }
        let n = UInt32(1 << logN)
        let invN = m31Inverse(M31(v: n))
        let buf = createM31Buffer([invN])!
        invNCache[logN] = buf
        return buf
    }

    private func createM31Buffer(_ data: [M31]) -> MTLBuffer? {
        let byteCount = data.count * MemoryLayout<M31>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            return nil
        }
        data.withUnsafeBytes { src in
            memcpy(buf.contents(), src.baseAddress!, byteCount)
        }
        return buf
    }

    // MARK: - Forward Circle NTT (DIT)

    /// Forward NTT: coefficients -> evaluations on the circle domain.
    /// Layers processed bottom-up: layer k-1 first (smallest blocks), then up to layer 0.
    /// But for DIT, we go: layer k-1 (block=2), ..., layer 1 (block=N/2), then layer 0 (block=N).
    /// Wait - in our formulation, layer 0 is the y-twiddle layer and is the LAST layer in forward.
    /// Forward order: layers k-1 down to 1 (x-twiddles), then layer 0 (y-twiddles).
    public func ntt(data: MTLBuffer, logN: Int) throws {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let twiddles = getForwardTwiddles(logN: logN)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n
        let enc = cmdBuf.makeComputeCommandEncoder()!
        let tgSize = min(256, Int(butterflyFunction.maxTotalThreadsPerThreadgroup))

        // Layers k-1 down to 1: x-twiddle butterflies
        // In our twiddle layout, layer l's twiddles start at offset l * (N/2)
        for layer in stride(from: logN - 1, through: 1, by: -1) {
            if layer > 1 { enc.memoryBarrier(scope: .buffers) }
            let stage = UInt32(logN - 1 - layer)  // stage in butterfly terms (block size = 2^(stage+1))
            // Actually, let me think about this differently.
            // The butterfly at 'layer' l processes blocks of size N / 2^l with half_block = N / 2^(l+1)
            // In the kernel, stage parameter = log2(half_block)
            var stageVal = UInt32(logN - 1 - layer)
            // Hmm, actually let me align with how twiddles are indexed.
            // For layer l, block_size = N >> l, half_block = N >> (l+1)
            // stage = log2(half_block) = logN - l - 1
            // twiddle_idx = local_idx * (n / block_size) = local_idx * 2^l
            // The twiddle array for this layer: x-twiddles at projection level (l-1)
            // But we store them packed. Let's use a different approach:
            // Store all twiddles for all stages in a flat array, stage s twiddles at offset s*N/2

            enc.setComputePipelineState(butterflyFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            // Twiddle buffer offset for this layer
            let twiddleOffset = layer * (nInt / 2) * MemoryLayout<M31>.stride
            enc.setBuffer(twiddles, offset: twiddleOffset, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            stageVal = UInt32(logN - 1 - layer)
            enc.setBytes(&stageVal, length: 4, index: 3)
            let numButterflies = nInt / 2
            enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        }

        // Layer 0: y-twiddle butterfly (the final/outermost layer in forward)
        if logN >= 1 {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(butterflyFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            // Layer 0 twiddles at offset 0
            enc.setBuffer(twiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var stageVal = UInt32(logN - 1)
            enc.setBytes(&stageVal, length: 4, index: 3)
            let numButterflies = nInt / 2
            enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        }

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - Inverse Circle NTT (DIF)

    /// Inverse NTT: evaluations -> coefficients.
    /// Layer 0 (y-twiddles) first, then layers 1..k-1 (x-twiddles), then scale by 1/N.
    public func intt(data: MTLBuffer, logN: Int) throws {
        let n = UInt32(1 << logN)
        let nInt = Int(n)
        let invTwiddles = getInverseTwiddles(logN: logN)
        let invN = getInvN(logN: logN)

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = n
        let enc = cmdBuf.makeComputeCommandEncoder()!
        let tgSize = min(256, Int(invButterflyFunction.maxTotalThreadsPerThreadgroup))

        // Layer 0: y-twiddle DIF butterfly
        if logN >= 1 {
            enc.setComputePipelineState(invButterflyFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            enc.setBuffer(invTwiddles, offset: 0, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var stageVal = UInt32(logN - 1)
            enc.setBytes(&stageVal, length: 4, index: 3)
            let numButterflies = nInt / 2
            enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        }

        // Layers 1..k-1: x-twiddle DIF butterflies
        for layer in 1..<logN {
            enc.memoryBarrier(scope: .buffers)
            enc.setComputePipelineState(invButterflyFunction)
            enc.setBuffer(data, offset: 0, index: 0)
            let twiddleOffset = layer * (nInt / 2) * MemoryLayout<M31>.stride
            enc.setBuffer(invTwiddles, offset: twiddleOffset, index: 1)
            enc.setBytes(&nVal, length: 4, index: 2)
            var stageVal = UInt32(logN - 1 - layer)
            enc.setBytes(&stageVal, length: 4, index: 3)
            let numButterflies = nInt / 2
            enc.dispatchThreads(MTLSize(width: numButterflies, height: 1, depth: 1),
                              threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        }

        // Scale by 1/N
        enc.memoryBarrier(scope: .buffers)
        enc.setComputePipelineState(scaleFunction)
        enc.setBuffer(data, offset: 0, index: 0)
        enc.setBuffer(invN, offset: 0, index: 1)
        enc.setBytes(&nVal, length: 4, index: 2)
        let tgScale = min(256, Int(scaleFunction.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: nInt, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: tgScale, height: 1, depth: 1))

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }
    }

    // MARK: - High-level API

    public func ntt(_ input: [M31]) throws -> [M31] {
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Circle NTT size must be power of 2")
        let logN = Int(log2(Double(n)))
        let stride = MemoryLayout<M31>.stride
        if n > cachedDataBufElements {
            guard let buf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create data buffer")
            }
            cachedDataBuf = buf
            cachedDataBufElements = n
        }
        let dataBuf = cachedDataBuf!
        input.withUnsafeBytes { src in
            memcpy(dataBuf.contents(), src.baseAddress!, n * stride)
        }
        try ntt(data: dataBuf, logN: logN)
        let ptr = dataBuf.contents().bindMemory(to: M31.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    public func intt(_ input: [M31]) throws -> [M31] {
        let n = input.count
        precondition(n > 0 && (n & (n - 1)) == 0, "Circle NTT size must be power of 2")
        let logN = Int(log2(Double(n)))
        let stride = MemoryLayout<M31>.stride
        if n > cachedDataBufElements {
            guard let buf = device.makeBuffer(length: n * stride, options: .storageModeShared) else {
                throw MSMError.gpuError("Failed to create data buffer")
            }
            cachedDataBuf = buf
            cachedDataBufElements = n
        }
        let dataBuf = cachedDataBuf!
        input.withUnsafeBytes { src in
            memcpy(dataBuf.contents(), src.baseAddress!, n * stride)
        }
        try intt(data: dataBuf, logN: logN)
        let ptr = dataBuf.contents().bindMemory(to: M31.self, capacity: n)
        return Array(UnsafeBufferPointer(start: ptr, count: n))
    }

    // MARK: - CPU Reference Implementation

    /// CPU reference Circle FFT: coefficients -> evaluations
    public static func cpuNTT(_ input: [M31], logN: Int) -> [M31] {
        let n = input.count
        precondition(n == 1 << logN)
        var data = input

        let domain = circleCosetDomain(logN: logN)

        // Precompute x-twiddle levels
        var xsLevels: [[M31]] = []
        var xs = (0..<n/2).map { domain[$0].x }
        for _ in 1..<logN {
            xsLevels.append(xs)
            let halfLen = xs.count / 2
            var newXs = [M31](repeating: M31.zero, count: halfLen)
            for i in 0..<halfLen {
                // Squaring map: 2x^2 - 1
                newXs[i] = m31Sub(m31Add(m31Sqr(xs[i]), m31Sqr(xs[i])), M31.one)
            }
            xs = newXs
        }

        // Forward: layers k-1 down to 1 (x-twiddle DIT)
        for layerIdx in stride(from: logN - 1, through: 1, by: -1) {
            let blockSize = n >> layerIdx
            let halfBlock = blockSize >> 1
            let twXs = xsLevels[layerIdx - 1]

            for blockStart in stride(from: 0, to: n, by: blockSize) {
                for j in 0..<halfBlock {
                    let i0 = blockStart + j
                    let i1 = i0 + halfBlock
                    let tw = twXs[j]
                    let u = data[i0]
                    let v = data[i1]
                    let twv = m31Mul(tw, v)
                    data[i0] = m31Add(u, twv)
                    data[i1] = m31Sub(u, twv)
                }
            }
        }

        // Layer 0: y-twiddle DIT butterfly
        if logN >= 1 {
            let half = n / 2
            for i in 0..<half {
                let tw = domain[i].y
                let u = data[i]
                let v = data[i + half]
                let twv = m31Mul(tw, v)
                data[i] = m31Add(u, twv)
                data[i + half] = m31Sub(u, twv)
            }
        }

        return data
    }

    /// CPU reference Circle IFFT: evaluations -> coefficients
    public static func cpuINTT(_ input: [M31], logN: Int) -> [M31] {
        let n = input.count
        precondition(n == 1 << logN)
        var data = input

        let domain = circleCosetDomain(logN: logN)

        // Layer 0: y-twiddle DIF butterfly
        if logN >= 1 {
            let half = n / 2
            for i in 0..<half {
                let twY = domain[i].y
                let invTwY = m31Inverse(twY)
                let v0 = data[i]
                let v1 = data[i + half]
                data[i] = m31Add(v0, v1)
                data[i + half] = m31Mul(m31Sub(v0, v1), invTwY)
            }
        }

        // Layers 1..k-1: x-twiddle DIF butterflies
        var xs = (0..<n/2).map { domain[$0].x }
        for layer in 1..<logN {
            let blockSize = n >> layer
            let halfBlock = blockSize >> 1

            for blockStart in stride(from: 0, to: n, by: blockSize) {
                for j in 0..<halfBlock {
                    let i0 = blockStart + j
                    let i1 = i0 + halfBlock
                    let tw = xs[j]
                    let invTw = m31Inverse(tw)
                    let v0 = data[i0]
                    let v1 = data[i1]
                    data[i0] = m31Add(v0, v1)
                    data[i1] = m31Mul(m31Sub(v0, v1), invTw)
                }
            }

            // Squaring map for next level
            let halfLen = xs.count / 2
            var newXs = [M31](repeating: M31.zero, count: halfLen)
            for i in 0..<halfLen {
                newXs[i] = m31Sub(m31Add(m31Sqr(xs[i]), m31Sqr(xs[i])), M31.one)
            }
            xs = newXs
        }

        // Scale by 1/N
        let invN = m31Inverse(M31(v: UInt32(n)))
        for i in 0..<n {
            data[i] = m31Mul(data[i], invN)
        }

        return data
    }
}

// MARK: - Twiddle precomputation

/// Precompute forward twiddles for Circle NTT.
/// Layout: [layer_0_twiddles (N/2)] [layer_1_twiddles (N/2)] ... [layer_{k-1}_twiddles (N/2)]
/// Layer 0: y-coordinates of domain points (padded to N/2)
/// Layer l (l>=1): x-coordinates at projection level l-1 (repeated to fill N/2)
public func circlePrecomputeForwardTwiddles(logN: Int) -> [M31] {
    let n = 1 << logN
    let half = n / 2
    let domain = circleCosetDomain(logN: logN)

    var allTwiddles = [M31]()
    allTwiddles.reserveCapacity(logN * half)

    // Layer 0: y-coordinate twiddles
    // For the butterfly with stage = logN-1, half_block = N/2
    // twiddle_idx = local_idx * (n / block_size) = local_idx * 1 = local_idx
    // So we need tw[0..N/2-1] = y-coords of domain points
    var layer0 = [M31](repeating: M31.zero, count: half)
    for i in 0..<half {
        layer0[i] = domain[i].y
    }
    allTwiddles.append(contentsOf: layer0)

    // Layers 1..k-1: x-coordinate twiddles
    var xs = (0..<half).map { domain[$0].x }
    for layer in 1..<logN {
        // For layer l, stage = logN-1-l, half_block = n >> (l+1)
        // twiddle_idx = local_idx * (n / block_size) = local_idx * 2^l
        // We need tw[j * 2^l] for j = 0..half_block-1, but the kernel indexes
        // into a flat array. We need to fill N/2 entries.
        // Since the kernel does: tw_idx = local_idx * (n / block_size)
        // and block_size = n >> l, so n / block_size = 2^l
        // tw_idx ranges over {0, 2^l, 2*2^l, ...} up to (half_block-1)*2^l
        // = (n/(2*block_size)) * (n/block_size) ... this gets complicated.
        //
        // Simpler: just store the raw x-values at natural positions.
        // The kernel indexes with stride, so we need to store at stride positions.

        let stride = 1 << layer
        var layerTw = [M31](repeating: M31.zero, count: half)
        let numValues = half / stride
        for j in 0..<numValues {
            if j < xs.count {
                layerTw[j * stride] = xs[j]
            }
        }
        allTwiddles.append(contentsOf: layerTw)

        // Apply squaring map for next level
        let halfLen = xs.count / 2
        var newXs = [M31](repeating: M31.zero, count: halfLen)
        for i in 0..<halfLen {
            newXs[i] = m31Sub(m31Add(m31Sqr(xs[i]), m31Sqr(xs[i])), M31.one)
        }
        xs = newXs
    }

    return allTwiddles
}

/// Precompute inverse twiddles for Circle IFFT.
/// Same layout as forward, but with inverted twiddle values.
public func circlePrecomputeInverseTwiddles(logN: Int) -> [M31] {
    let n = 1 << logN
    let half = n / 2
    let domain = circleCosetDomain(logN: logN)

    var allTwiddles = [M31]()
    allTwiddles.reserveCapacity(logN * half)

    // Layer 0: inverse y-coordinate twiddles
    var layer0 = [M31](repeating: M31.zero, count: half)
    for i in 0..<half {
        layer0[i] = m31Inverse(domain[i].y)
    }
    allTwiddles.append(contentsOf: layer0)

    // Layers 1..k-1: inverse x-coordinate twiddles
    var xs = (0..<half).map { domain[$0].x }
    for layer in 1..<logN {
        let stride = 1 << layer
        var layerTw = [M31](repeating: M31.zero, count: half)
        let numValues = half / stride
        for j in 0..<numValues {
            if j < xs.count {
                layerTw[j * stride] = m31Inverse(xs[j])
            }
        }
        allTwiddles.append(contentsOf: layerTw)

        let halfLen = xs.count / 2
        var newXs = [M31](repeating: M31.zero, count: halfLen)
        for i in 0..<halfLen {
            newXs[i] = m31Sub(m31Add(m31Sqr(xs[i]), m31Sqr(xs[i])), M31.one)
        }
        xs = newXs
    }

    return allTwiddles
}
