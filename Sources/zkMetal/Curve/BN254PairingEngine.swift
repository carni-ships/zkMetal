// BN254 GPU Pairing Engine — Metal-accelerated batch Miller loop for Groth16 verification
//
// Strategy: GPU computes N independent Miller loops in parallel (one per thread),
// then CPU multiplies results and performs final exponentiation.
// For Groth16 (4 pairings): 4 parallel GPU Miller loops + CPU product + CPU final exp.
//
// The Miller loop is the bottleneck (~65% of pairing time) and parallelizes well.
// Final exponentiation is done once on the product (shared across all pairings).
//
// Optimizations:
// - Projective Miller loop: eliminates all fp2_inverse calls (~85 per loop)
// - Grow-only buffer caching: avoids per-call Metal buffer allocation
// - Single command buffer: all dispatches in one CB
// - Sparse Fp12 multiplication for line evaluations

import Foundation
import Metal

public enum PairingError: Error {
    case noGPU
    case noCommandQueue
    case compilationFailed(String)
    case missingKernel(String)
    case executionFailed(String)
}

public class BN254PairingEngine {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    private let batchMillerFunction: MTLComputePipelineState

    // Grow-only cached buffers to avoid per-call allocation
    private var g1Buffer: MTLBuffer?
    private var g1BufferSize: Int = 0
    private var g2Buffer: MTLBuffer?
    private var g2BufferSize: Int = 0
    private var resultBuffer: MTLBuffer?
    private var resultBufferSize: Int = 0
    private var countBuffer: MTLBuffer?

    // Profiling counters (accumulated across calls, reset manually)
    public var profilingEnabled: Bool = false
    public private(set) var profileMillerGPUMs: Double = 0
    public private(set) var profileFinalExpMs: Double = 0
    public private(set) var profileDataPackMs: Double = 0
    public private(set) var profileDataUnpackMs: Double = 0
    public private(set) var profileCallCount: Int = 0

    public func resetProfiling() {
        profileMillerGPUMs = 0
        profileFinalExpMs = 0
        profileDataPackMs = 0
        profileDataUnpackMs = 0
        profileCallCount = 0
    }

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw PairingError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw PairingError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try BN254PairingEngine.compileShaders(device: device)

        guard let fn = library.makeFunction(name: "batch_miller_loop") else {
            throw PairingError.missingKernel("batch_miller_loop")
        }
        self.batchMillerFunction = try device.makeComputePipelineState(function: fn)

        // Pre-allocate count buffer (always 4 bytes)
        self.countBuffer = device.makeBuffer(length: 4, options: .storageModeShared)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()

        let fpSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fp.metal", encoding: .utf8)
        let pairingSource = try String(contentsOfFile: shaderDir + "/pairing/bn254_pairing.metal", encoding: .utf8)

        func stripIncludes(_ s: String) -> String {
            s.split(separator: "\n").filter { !$0.contains("#include") }.joined(separator: "\n")
        }
        func stripGuards(_ s: String) -> String {
            s.replacingOccurrences(of: "#ifndef BN254_FP_METAL", with: "")
             .replacingOccurrences(of: "#define BN254_FP_METAL", with: "")
             .replacingOccurrences(of: "#endif // BN254_FP_METAL", with: "")
        }

        let combined = stripGuards(fpSource) + "\n" + stripGuards(stripIncludes(pairingSource))

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        do {
            return try device.makeLibrary(source: combined, options: options)
        } catch {
            throw PairingError.compilationFailed(error.localizedDescription)
        }
    }

    // MARK: - Grow-only buffer management

    private func ensureG1Buffer(size: Int) {
        if g1BufferSize < size {
            g1Buffer = device.makeBuffer(length: size, options: .storageModeShared)
            g1BufferSize = size
        }
    }

    private func ensureG2Buffer(size: Int) {
        if g2BufferSize < size {
            g2Buffer = device.makeBuffer(length: size, options: .storageModeShared)
            g2BufferSize = size
        }
    }

    private func ensureResultBuffer(size: Int) {
        if resultBufferSize < size {
            resultBuffer = device.makeBuffer(length: size, options: .storageModeShared)
            resultBufferSize = size
        }
    }

    // MARK: - GPU Data Layout

    private func packG1Into(_ p: PointAffine, buffer: UnsafeMutableRawPointer, offset: Int) {
        let limbs = [p.x.v.0, p.x.v.1, p.x.v.2, p.x.v.3, p.x.v.4, p.x.v.5, p.x.v.6, p.x.v.7,
                     p.y.v.0, p.y.v.1, p.y.v.2, p.y.v.3, p.y.v.4, p.y.v.5, p.y.v.6, p.y.v.7]
        let dst = buffer.advanced(by: offset).bindMemory(to: UInt32.self, capacity: 16)
        for i in 0..<16 { dst[i] = limbs[i] }
    }

    private func packG2Into(_ q: G2AffinePoint, buffer: UnsafeMutableRawPointer, offset: Int) {
        let limbs = [q.x.c0.v.0, q.x.c0.v.1, q.x.c0.v.2, q.x.c0.v.3,
                     q.x.c0.v.4, q.x.c0.v.5, q.x.c0.v.6, q.x.c0.v.7,
                     q.x.c1.v.0, q.x.c1.v.1, q.x.c1.v.2, q.x.c1.v.3,
                     q.x.c1.v.4, q.x.c1.v.5, q.x.c1.v.6, q.x.c1.v.7,
                     q.y.c0.v.0, q.y.c0.v.1, q.y.c0.v.2, q.y.c0.v.3,
                     q.y.c0.v.4, q.y.c0.v.5, q.y.c0.v.6, q.y.c0.v.7,
                     q.y.c1.v.0, q.y.c1.v.1, q.y.c1.v.2, q.y.c1.v.3,
                     q.y.c1.v.4, q.y.c1.v.5, q.y.c1.v.6, q.y.c1.v.7]
        let dst = buffer.advanced(by: offset).bindMemory(to: UInt32.self, capacity: 32)
        for i in 0..<32 { dst[i] = limbs[i] }
    }

    private func unpackFp12(data: UnsafeRawPointer) -> Fp12 {
        let ptr = data.bindMemory(to: UInt32.self, capacity: 96)
        func readFp(_ offset: Int) -> Fp {
            Fp(v: (ptr[offset], ptr[offset+1], ptr[offset+2], ptr[offset+3],
                   ptr[offset+4], ptr[offset+5], ptr[offset+6], ptr[offset+7]))
        }
        func readFp2(_ offset: Int) -> Fp2 {
            Fp2(c0: readFp(offset), c1: readFp(offset+8))
        }
        func readFp6(_ offset: Int) -> Fp6 {
            Fp6(c0: readFp2(offset), c1: readFp2(offset+16), c2: readFp2(offset+32))
        }
        return Fp12(c0: readFp6(0), c1: readFp6(48))
    }

    // MARK: - Public API

    /// Run N Miller loops in parallel on GPU. Returns array of Fp12 results.
    public func batchMillerLoop(pairs: [(PointAffine, G2AffinePoint)]) throws -> [Fp12] {
        let n = pairs.count
        if n == 0 { return [] }

        let g1Size = n * 64   // 2 Fp = 64 bytes per G1 point
        let g2Size = n * 128  // 4 Fp = 128 bytes per G2 point
        let resSize = n * 384 // 12 Fp = 384 bytes per Fp12

        var t0 = CFAbsoluteTimeGetCurrent()

        // Ensure cached buffers are large enough
        ensureG1Buffer(size: g1Size)
        ensureG2Buffer(size: g2Size)
        ensureResultBuffer(size: resSize)

        guard let g1Buf = g1Buffer, let g2Buf = g2Buffer,
              let resBuf = resultBuffer, let cntBuf = countBuffer else {
            throw PairingError.executionFailed("Failed to allocate buffers")
        }

        // Pack data directly into cached buffers
        let g1Ptr = g1Buf.contents()
        let g2Ptr = g2Buf.contents()
        for i in 0..<n {
            packG1Into(pairs[i].0, buffer: g1Ptr, offset: i * 64)
            packG2Into(pairs[i].1, buffer: g2Ptr, offset: i * 128)
        }
        cntBuf.contents().storeBytes(of: UInt32(n), as: UInt32.self)

        if profilingEnabled {
            profileDataPackMs += (CFAbsoluteTimeGetCurrent() - t0) * 1000
        }

        t0 = CFAbsoluteTimeGetCurrent()

        // Single command buffer for the dispatch
        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            throw PairingError.executionFailed("Failed to create command buffer")
        }

        encoder.setComputePipelineState(batchMillerFunction)
        encoder.setBuffer(g1Buf, offset: 0, index: 0)
        encoder.setBuffer(g2Buf, offset: 0, index: 1)
        encoder.setBuffer(resBuf, offset: 0, index: 2)
        encoder.setBuffer(cntBuf, offset: 0, index: 3)

        let tgSize = MTLSize(width: min(n, batchMillerFunction.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let gridSize = MTLSize(width: n, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if profilingEnabled {
            profileMillerGPUMs += (CFAbsoluteTimeGetCurrent() - t0) * 1000
        }

        if let error = cmdBuf.error {
            throw PairingError.executionFailed("GPU error: \(error)")
        }

        t0 = CFAbsoluteTimeGetCurrent()

        var results = [Fp12]()
        results.reserveCapacity(n)
        let ptr = resBuf.contents()
        for i in 0..<n {
            results.append(unpackFp12(data: ptr + i * 384))
        }

        if profilingEnabled {
            profileDataUnpackMs += (CFAbsoluteTimeGetCurrent() - t0) * 1000
            profileCallCount += 1
        }

        return results
    }

    /// Compute N independent pairings: GPU parallel Miller loops + CPU final exp per result.
    public func batchPairing(pairs: [(PointAffine, G2AffinePoint)]) throws -> [Fp12] {
        let millerResults = try batchMillerLoop(pairs: pairs)
        let t0 = CFAbsoluteTimeGetCurrent()
        let results = millerResults.map { bn254FinalExponentiation($0) }
        if profilingEnabled {
            profileFinalExpMs += (CFAbsoluteTimeGetCurrent() - t0) * 1000
        }
        return results
    }

    /// Multi-Miller pairing: N parallel Miller loops on GPU, CPU product, single CPU final exp.
    /// Optimal for Groth16 verification where you need prod_i e(Pi, Qi).
    public func multiMillerPairing(pairs: [(PointAffine, G2AffinePoint)]) throws -> Fp12 {
        let millerResults = try batchMillerLoop(pairs: pairs)
        let t0 = CFAbsoluteTimeGetCurrent()
        var product = Fp12.one
        for m in millerResults {
            product = fp12Mul(product, m)
        }
        let result = bn254FinalExponentiation(product)
        if profilingEnabled {
            profileFinalExpMs += (CFAbsoluteTimeGetCurrent() - t0) * 1000
        }
        return result
    }

    /// Pairing check: verify prod_i e(Pi, Qi) == 1.
    /// GPU parallel Miller loops + CPU product + CPU final exp + equality check.
    public func pairingCheck(pairs: [(PointAffine, G2AffinePoint)]) throws -> Bool {
        let result = try multiMillerPairing(pairs: pairs)
        return fp12Equal(result, .one)
    }

    /// Print profiling summary to stderr.
    public func printProfile() {
        guard profileCallCount > 0 else {
            fputs("  No profiling data collected.\n", stderr)
            return
        }
        let total = profileDataPackMs + profileMillerGPUMs + profileDataUnpackMs + profileFinalExpMs
        fputs(String(format: "  Pairing profile (%d calls, %.1fms total):\n", profileCallCount, total), stderr)
        fputs(String(format: "    Data pack:    %7.2fms (%4.1f%%)\n", profileDataPackMs, profileDataPackMs / total * 100), stderr)
        fputs(String(format: "    Miller GPU:   %7.2fms (%4.1f%%)\n", profileMillerGPUMs, profileMillerGPUMs / total * 100), stderr)
        fputs(String(format: "    Data unpack:  %7.2fms (%4.1f%%)\n", profileDataUnpackMs, profileDataUnpackMs / total * 100), stderr)
        fputs(String(format: "    Final exp:    %7.2fms (%4.1f%%)\n", profileFinalExpMs, profileFinalExpMs / total * 100), stderr)
    }
}
