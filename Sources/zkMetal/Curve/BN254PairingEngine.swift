// BN254 GPU Pairing Engine — Metal-accelerated batch Miller loop for Groth16 verification
//
// Strategy: GPU computes N independent Miller loops in parallel (one per thread),
// then CPU multiplies results and performs final exponentiation.
// For Groth16 (4 pairings): 4 parallel GPU Miller loops + CPU product + CPU final exp.
//
// The Miller loop is the bottleneck (~65% of pairing time) and parallelizes well.
// Final exponentiation is done once on the product (shared across all pairings).

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

    // MARK: - GPU Data Layout

    private func packG1(_ p: PointAffine) -> [UInt8] {
        var data = [UInt8]()
        data.append(contentsOf: fpToRawBytes(p.x))
        data.append(contentsOf: fpToRawBytes(p.y))
        return data
    }

    private func packG2(_ q: G2AffinePoint) -> [UInt8] {
        var data = [UInt8]()
        data.append(contentsOf: fpToRawBytes(q.x.c0))
        data.append(contentsOf: fpToRawBytes(q.x.c1))
        data.append(contentsOf: fpToRawBytes(q.y.c0))
        data.append(contentsOf: fpToRawBytes(q.y.c1))
        return data
    }

    private func fpToRawBytes(_ a: Fp) -> [UInt8] {
        let limbs = [a.v.0, a.v.1, a.v.2, a.v.3, a.v.4, a.v.5, a.v.6, a.v.7]
        var bytes = [UInt8](repeating: 0, count: 32)
        for i in 0..<8 {
            bytes[i*4+0] = UInt8(limbs[i] & 0xFF)
            bytes[i*4+1] = UInt8((limbs[i] >> 8) & 0xFF)
            bytes[i*4+2] = UInt8((limbs[i] >> 16) & 0xFF)
            bytes[i*4+3] = UInt8((limbs[i] >> 24) & 0xFF)
        }
        return bytes
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

        var g1Data = [UInt8]()
        var g2Data = [UInt8]()
        for (p, q) in pairs {
            g1Data.append(contentsOf: packG1(p))
            g2Data.append(contentsOf: packG2(q))
        }

        guard let g1Buf = device.makeBuffer(bytes: g1Data, length: g1Data.count, options: .storageModeShared),
              let g2Buf = device.makeBuffer(bytes: g2Data, length: g2Data.count, options: .storageModeShared),
              let resultBuf = device.makeBuffer(length: n * 384, options: .storageModeShared),
              let countBuf = device.makeBuffer(length: 4, options: .storageModeShared) else {
            throw PairingError.executionFailed("Failed to allocate buffers")
        }

        countBuf.contents().storeBytes(of: UInt32(n), as: UInt32.self)

        guard let cmdBuf = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuf.makeComputeCommandEncoder() else {
            throw PairingError.executionFailed("Failed to create command buffer")
        }

        encoder.setComputePipelineState(batchMillerFunction)
        encoder.setBuffer(g1Buf, offset: 0, index: 0)
        encoder.setBuffer(g2Buf, offset: 0, index: 1)
        encoder.setBuffer(resultBuf, offset: 0, index: 2)
        encoder.setBuffer(countBuf, offset: 0, index: 3)

        let tgSize = MTLSize(width: min(n, batchMillerFunction.maxTotalThreadsPerThreadgroup), height: 1, depth: 1)
        let gridSize = MTLSize(width: n, height: 1, depth: 1)
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        encoder.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw PairingError.executionFailed("GPU error: \(error)")
        }

        var results = [Fp12]()
        let ptr = resultBuf.contents()
        for i in 0..<n {
            results.append(unpackFp12(data: ptr + i * 384))
        }
        return results
    }

    /// Compute N independent pairings: GPU parallel Miller loops + CPU final exp per result.
    public func batchPairing(pairs: [(PointAffine, G2AffinePoint)]) throws -> [Fp12] {
        let millerResults = try batchMillerLoop(pairs: pairs)
        return millerResults.map { bn254FinalExponentiation($0) }
    }

    /// Multi-Miller pairing: N parallel Miller loops on GPU, CPU product, single CPU final exp.
    /// Optimal for Groth16 verification where you need prod_i e(Pi, Qi).
    public func multiMillerPairing(pairs: [(PointAffine, G2AffinePoint)]) throws -> Fp12 {
        let millerResults = try batchMillerLoop(pairs: pairs)
        var product = Fp12.one
        for m in millerResults {
            product = fp12Mul(product, m)
        }
        return bn254FinalExponentiation(product)
    }

    /// Pairing check: verify prod_i e(Pi, Qi) == 1.
    /// GPU parallel Miller loops + CPU product + CPU final exp + equality check.
    public func pairingCheck(pairs: [(PointAffine, G2AffinePoint)]) throws -> Bool {
        let result = try multiMillerPairing(pairs: pairs)
        return fp12Equal(result, .one)
    }
}
