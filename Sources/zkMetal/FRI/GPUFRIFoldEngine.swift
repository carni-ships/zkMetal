// GPUFRIFoldEngine — Dedicated GPU-accelerated FRI folding engine for STARK provers
//
// Core operation: given evaluations on a domain, fold them using a random
// challenge to produce evaluations on a half-sized domain. This is the
// main bottleneck in STARK provers (Plonky2, Plonky3, Stwo, etc.).
//
// Unlike the general-purpose FRIEngine (which uses inverse twiddles),
// this engine works with explicit domain representations, supporting
// coset domains, shifted domains, and arbitrary evaluation points.
//
// Formula: result[i] = (evals[i] + evals[i + n/2]) + challenge * (evals[i] - evals[i + n/2]) / domain[i]
//
// The domain inverses (1/domain[i]) are precomputed on GPU and cached
// between rounds. Intermediate results stay on GPU across fold rounds
// to minimize PCIe/unified-memory round-trips.

import Foundation
import Metal
import NeonFieldOps

public class GPUFRIFoldEngine {
    public static let version = Versions.friFold

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Pipeline states
    private let foldKernel: MTLComputePipelineState
    private let foldFused2Kernel: MTLComputePipelineState
    private let domainInverseKernel: MTLComputePipelineState

    // Cached domain inverse buffers: [logN] -> MTLBuffer
    private var domainInvCache: [Int: MTLBuffer] = [:]

    // Ping-pong buffers for multi-round folding (kept on GPU)
    private var pingBuf: MTLBuffer?
    private var pongBuf: MTLBuffer?
    private var pingPongBytes: Int = 0

    // CPU fallback threshold
    public static let cpuFallbackThreshold = 512

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
        self.tuning = TuningManager.shared.config(device: device)

        let library = try GPUFRIFoldEngine.compileShaders(device: device)

        guard let foldFn = library.makeFunction(name: "fri_fold_kernel"),
              let fused2Fn = library.makeFunction(name: "fri_fold_fused2_kernel"),
              let invFn = library.makeFunction(name: "fri_domain_inverse_kernel") else {
            throw MSMError.missingKernel
        }

        self.foldKernel = try device.makeComputePipelineState(function: foldFn)
        self.foldFused2Kernel = try device.makeComputePipelineState(function: fused2Fn)
        self.domainInverseKernel = try device.makeComputePipelineState(function: invFn)
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let foldSource = try String(contentsOfFile: shaderDir + "/fri/fri_fold.metal", encoding: .utf8)

        // Strip #include directives (we inline dependencies)
        let cleanFold = foldSource.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let combined = cleanFr + "\n" + cleanFold
        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: combined, options: options)
    }

    // MARK: - Domain Inverse Precomputation

    /// Precompute domain element inverses on GPU: domain_inv[i] = 1/domain[i].
    /// Returns an MTLBuffer of Fr elements in Montgomery form.
    public func precomputeDomainInverses(domain: [Fr]) -> MTLBuffer {
        let count = domain.count
        let stride = MemoryLayout<Fr>.stride

        // Upload domain to GPU
        let domainBuf = device.makeBuffer(
            bytes: domain, length: count * stride,
            options: .storageModeShared)!

        // Small domains: CPU fallback with Montgomery batch inversion
        if count < GPUFRIFoldEngine.cpuFallbackThreshold {
            var invPrefix = [Fr](repeating: Fr.one, count: count)
            for i in 1..<count {
                invPrefix[i] = domain[i - 1] == Fr.zero ? invPrefix[i - 1] : frMul(invPrefix[i - 1], domain[i - 1])
            }
            let invLast = domain[count - 1] == Fr.zero ? invPrefix[count - 1] : frMul(invPrefix[count - 1], domain[count - 1])
            var invR = frInverse(invLast)
            var invs = [Fr](repeating: Fr.zero, count: count)
            for i in Swift.stride(from: count - 1, through: 0, by: -1) {
                if domain[i] != Fr.zero {
                    invs[i] = frMul(invR, invPrefix[i])
                    invR = frMul(invR, domain[i])
                }
            }
            let outBuf = device.makeBuffer(
                bytes: invs, length: count * stride,
                options: .storageModeShared)!
            return outBuf
        }

        let outBuf = device.makeBuffer(length: count * stride, options: .storageModeShared)!

        let cmdBuf = commandQueue.makeCommandBuffer()!
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(domainInverseKernel)
        enc.setBuffer(domainBuf, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        var countVal = UInt32(count)
        enc.setBytes(&countVal, length: 4, index: 2)

        let tg = min(tuning.friThreadgroupSize,
                     Int(domainInverseKernel.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        return outBuf
    }

    /// Precompute and cache domain inverses for a multiplicative domain of size 2^logN.
    /// Domain elements are omega^0, omega^1, ..., omega^{n/2-1} where omega is
    /// the n-th root of unity. We only need inverses for the first half.
    private func getCachedDomainInverses(logN: Int) -> MTLBuffer {
        if let cached = domainInvCache[logN] { return cached }

        let n = 1 << logN
        let half = n / 2

        // Build domain: omega^i for i in [0, half)
        let omega = rootOfUnity(logN: logN)
        var domain = [Fr](repeating: Fr.one, count: half)
        var w = Fr.one
        for i in 0..<half {
            domain[i] = w
            w = frMul(w, omega)
        }

        let buf = precomputeDomainInverses(domain: domain)
        domainInvCache[logN] = buf
        return buf
    }

    // MARK: - Single Fold Round

    /// Perform one FRI fold round on GPU.
    /// Input: evals buffer of size n (Fr elements).
    /// Output: new MTLBuffer of size n/2.
    /// domain_inv: precomputed 1/domain[i] for i in [0, n/2).
    /// challenge: the random folding challenge.
    public func fold(evals: MTLBuffer, domainInv: MTLBuffer,
                     challenge: Fr, n: Int) throws -> MTLBuffer {
        let half = n / 2
        let stride = MemoryLayout<Fr>.stride

        // CPU fallback for small inputs
        if n < GPUFRIFoldEngine.cpuFallbackThreshold {
            return try cpuFold(evals: evals, domainInv: domainInv,
                               challenge: challenge, n: n)
        }

        let outBuf = device.makeBuffer(length: half * stride, options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }

        var nVal = UInt32(n)
        var challengeVal = challenge
        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(foldKernel)
        enc.setBuffer(evals, offset: 0, index: 0)
        enc.setBuffer(outBuf, offset: 0, index: 1)
        enc.setBuffer(domainInv, offset: 0, index: 2)
        enc.setBytes(&challengeVal, length: stride, index: 3)
        enc.setBytes(&nVal, length: 4, index: 4)

        let tg = min(tuning.friThreadgroupSize,
                     Int(foldKernel.maxTotalThreadsPerThreadgroup))
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

    /// Convenience: fold using a multiplicative domain (roots of unity).
    /// Automatically precomputes and caches domain inverses.
    public func fold(evals: MTLBuffer, logN: Int, challenge: Fr) throws -> MTLBuffer {
        let n = 1 << logN
        let domainInv = getCachedDomainInverses(logN: logN)
        return try fold(evals: evals, domainInv: domainInv, challenge: challenge, n: n)
    }

    // MARK: - Multi-Round Fold (GPU-Resident)

    /// Fold multiple rounds in a single command buffer, keeping data on GPU
    /// between rounds. Returns the final folded buffer.
    /// challenges: one challenge per sub-fold round (e.g., 3 challenges for fold-by-8)
    /// logN: log2 of initial domain size
    public func foldMultiRound(evals: MTLBuffer, logN: Int,
                                challenges: [Fr]) throws -> MTLBuffer {
        let numRounds = challenges.count
        precondition(numRounds > 0 && numRounds <= logN)

        let stride = MemoryLayout<Fr>.stride
        var currentN = 1 << logN

        // Ensure ping-pong buffers are large enough for the first output
        let maxOutSize = (currentN / 2) * stride
        if pingPongBytes < maxOutSize {
            pingBuf = device.makeBuffer(length: maxOutSize, options: .storageModeShared)
            pongBuf = device.makeBuffer(length: maxOutSize, options: .storageModeShared)
            pingPongBytes = maxOutSize
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!

        var currentBuf = evals
        var usePing = true

        for round in 0..<numRounds {
            let half = currentN / 2
            let domainInv = getCachedDomainInverses(logN: logN - round)
            let outBuf = usePing ? pingBuf! : pongBuf!

            var nVal = UInt32(currentN)
            var challengeVal = challenges[round]
            enc.setComputePipelineState(foldKernel)
            enc.setBuffer(currentBuf, offset: 0, index: 0)
            enc.setBuffer(outBuf, offset: 0, index: 1)
            enc.setBuffer(domainInv, offset: 0, index: 2)
            enc.setBytes(&challengeVal, length: stride, index: 3)
            enc.setBytes(&nVal, length: 4, index: 4)

            let tg = min(tuning.friThreadgroupSize,
                         Int(foldKernel.maxTotalThreadsPerThreadgroup))
            enc.dispatchThreads(MTLSize(width: half, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))

            if round < numRounds - 1 {
                enc.memoryBarrier(scope: .buffers)
            }

            currentBuf = outBuf
            currentN = half
            usePing = !usePing
        }

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()
        if let error = cmdBuf.error {
            throw MSMError.gpuError(error.localizedDescription)
        }

        return currentBuf
    }

    // MARK: - Full FRI Commit (All Rounds)

    /// Fold through all FRI layers, keeping intermediate results on GPU.
    /// Returns all intermediate layer evaluations (for Merkle commitment).
    ///
    /// evals: initial evaluations (size n = 2^logN)
    /// domain: full domain elements (size n). For round k, domain elements
    ///         for the folded domain are computed by squaring.
    /// challenges: one challenge per fold round
    ///
    /// Returns: array of Fr arrays, one per round (including the initial evals).
    ///          layers[0] = input evals, layers[k] = result after k folds.
    public func foldAllRounds(evals: [Fr], domain: [Fr],
                               challenges: [Fr]) throws -> [[Fr]] {
        let n = evals.count
        precondition(n > 1 && (n & (n - 1)) == 0, "Size must be power of 2")
        precondition(domain.count == n, "Domain size must match evals")
        precondition(challenges.count > 0, "Need at least one challenge")

        let stride = MemoryLayout<Fr>.stride
        var layers: [[Fr]] = [evals]

        // Upload initial evals to GPU
        var currentBuf = device.makeBuffer(
            bytes: evals, length: n * stride,
            options: .storageModeShared)!
        var currentN = n

        // Precompute domain inverses for each round
        // Round 0: domain inverses for first half of original domain
        // Round k: domain is the "squared" domain from round k-1
        var currentDomain = domain

        for round in 0..<challenges.count {
            let half = currentN / 2

            // Compute domain inverses for this round's domain (first half)
            let halfDomain = Array(currentDomain[0..<half])
            let domainInvBuf = precomputeDomainInverses(domain: halfDomain)

            // Fold on GPU
            let foldedBuf = try fold(evals: currentBuf, domainInv: domainInvBuf,
                                      challenge: challenges[round], n: currentN)

            // Read back for layer storage
            let ptr = foldedBuf.contents().bindMemory(to: Fr.self, capacity: half)
            let foldedArray = Array(UnsafeBufferPointer(start: ptr, count: half))
            layers.append(foldedArray)

            // Prepare next round: square the domain for the folded domain
            if round < challenges.count - 1 {
                var nextDomain = [Fr](repeating: Fr.zero, count: half)
                for i in 0..<half {
                    nextDomain[i] = frMul(currentDomain[i], currentDomain[i])
                }
                currentDomain = nextDomain
            }

            currentBuf = foldedBuf
            currentN = half
        }

        return layers
    }

    /// Convenience: foldAllRounds using a multiplicative domain (roots of unity).
    /// Automatically generates the domain from the n-th root of unity.
    public func foldAllRounds(evals: [Fr], logN: Int,
                               challenges: [Fr]) throws -> [[Fr]] {
        let n = 1 << logN
        precondition(evals.count == n)

        // Build multiplicative domain: omega^0, omega^1, ..., omega^{n-1}
        let omega = rootOfUnity(logN: logN)
        var domain = [Fr](repeating: Fr.one, count: n)
        var w = Fr.one
        for i in 0..<n {
            domain[i] = w
            w = frMul(w, omega)
        }

        return try foldAllRounds(evals: evals, domain: domain, challenges: challenges)
    }

    /// GPU-resident multi-round fold: folds evals through all challenges,
    /// keeping data on GPU between rounds. Returns only the final result buffer.
    /// More efficient than foldAllRounds when intermediate layers are not needed.
    public func foldToFinal(evals: MTLBuffer, logN: Int,
                             challenges: [Fr]) throws -> MTLBuffer {
        precondition(challenges.count <= logN, "Too many challenges for domain size")
        precondition(challenges.count > 0, "Need at least one challenge")

        var currentBuf = evals
        var currentLogN = logN

        for i in 0..<challenges.count {
            let n = 1 << currentLogN

            if n < GPUFRIFoldEngine.cpuFallbackThreshold {
                let domainInv = getCachedDomainInverses(logN: currentLogN)
                currentBuf = try cpuFold(evals: currentBuf, domainInv: domainInv,
                                          challenge: challenges[i], n: n)
            } else {
                let domainInv = getCachedDomainInverses(logN: currentLogN)
                currentBuf = try fold(evals: currentBuf, domainInv: domainInv,
                                       challenge: challenges[i], n: n)
            }

            currentLogN -= 1
        }

        return currentBuf
    }

    // MARK: - CPU Fallback

    private func cpuFold(evals: MTLBuffer, domainInv: MTLBuffer,
                          challenge: Fr, n: Int) throws -> MTLBuffer {
        let half = n / 2
        let stride = MemoryLayout<Fr>.stride

        let evalsPtr = evals.contents().bindMemory(to: Fr.self, capacity: n)
        let dinvPtr = domainInv.contents().bindMemory(to: Fr.self, capacity: half)

        var folded = [Fr](repeating: Fr.zero, count: half)
        folded.withUnsafeMutableBytes { fBuf in
            withUnsafeBytes(of: challenge) { cBuf in
                bn254_fr_fri_fold(
                    UnsafeRawPointer(evalsPtr).assumingMemoryBound(to: UInt64.self),
                    cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    UnsafeRawPointer(dinvPtr).assumingMemoryBound(to: UInt64.self),
                    fBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(half))
            }
        }

        guard let outBuf = device.makeBuffer(bytes: folded, length: half * stride,
                                              options: .storageModeShared) else {
            throw MSMError.gpuError("CPU fold: failed to allocate output")
        }
        return outBuf
    }

    // MARK: - Domain Helpers

    /// Compute the n-th root of unity for the BN254 scalar field.
    private func rootOfUnity(logN: Int) -> Fr {
        // Use the existing precomputeInverseTwiddles infrastructure to get omega.
        // The 2^k-th root of unity for BN254 Fr is well-known.
        // We reconstruct omega from the inverse twiddle: omega = frInverse(invTwiddle[1])
        // when invTwiddle[i] = omega^{-i}.
        // Alternatively, compute directly: omega_{2^k} = g^{(r-1)/2^k} where g is a generator.
        // BN254 Fr two-adicity is 28, primitive 2^28-th root of unity is known.
        //
        // For simplicity, reuse the existing precomputeInverseTwiddles and invert the first element.
        let invTwiddles = precomputeInverseTwiddles(logN: logN)
        // invTwiddles[1] = omega^{-1}, so omega = frInverse(invTwiddles[1])
        return frInverse(invTwiddles[1])
    }

    /// Clear cached domain inverse buffers to free GPU memory.
    public func clearCache() {
        domainInvCache.removeAll()
        pingBuf = nil
        pongBuf = nil
        pingPongBytes = 0
    }
}
