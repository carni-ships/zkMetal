// Binary FFT Engine — Additive FFT over binary tower fields
// Binius uses additive FFTs (not multiplicative NTT) because binary fields
// lack multiplicative subgroups of power-of-2 order.
//
// The butterfly is: (a, b) -> (a + b, b * beta)
// where beta comes from a basis of the subspace.
// Addition = XOR (free), so butterfly cost is dominated by one multiplication.
//
// Based on the "novel polynomial basis" approach from:
// - Lin, Chung, Han (2014) — efficient additive FFT
// - Binius (2024) — binary tower STARK construction
//
// CPU reference implementation + GPU-accelerated batch version.

import Foundation
import Metal

// MARK: - CPU Additive FFT

/// Additive FFT over GF(2^32) using the binary tower structure.
/// Input: polynomial coefficients in the "novel polynomial basis"
/// Output: evaluations at a subspace of GF(2^32)
public enum BinaryFFT {

    /// Generate a basis for a 2^logN-dimensional subspace of GF(2^32)
    /// Uses the standard basis: {1, x, x^2, x^4, ...} where x is the tower generator
    public static func subspaceBasis(logN: Int) -> [BinaryField32] {
        precondition(logN >= 1 && logN <= 32)
        var basis = [BinaryField32]()
        basis.reserveCapacity(logN)
        // Use the canonical basis: e_i has bit i set
        for i in 0..<logN {
            basis.append(BinaryField32(value: 1 << i))
        }
        return basis
    }

    /// Precompute twiddle factors for the additive FFT
    /// For each level l, the twiddle is basis[l]
    public static func twiddleFactors(logN: Int) -> [[BinaryField32]] {
        let basis = subspaceBasis(logN: logN)
        var twiddles = [[BinaryField32]]()
        twiddles.reserveCapacity(logN)
        for l in 0..<logN {
            let halfSize = 1 << (logN - 1 - l)
            var tw = [BinaryField32](repeating: .zero, count: halfSize)
            // All twiddles at this level use basis[logN - 1 - l]
            let beta = basis[logN - 1 - l]
            for j in 0..<halfSize {
                tw[j] = beta
            }
            twiddles.append(tw)
        }
        return twiddles
    }

    /// Forward additive FFT (in-place)
    /// Transforms n = 2^logN coefficients to evaluations
    public static func forward(data: inout [BinaryField32], logN: Int) {
        let n = 1 << logN
        precondition(data.count == n)

        let basis = subspaceBasis(logN: logN)

        // Butterfly network: logN stages
        // At stage l (0-indexed), block size = n >> l, half = block_size/2
        for l in 0..<logN {
            let blockSize = n >> l
            let half = blockSize >> 1
            let beta = basis[logN - 1 - l]

            var block = 0
            while block < n {
                for j in 0..<half {
                    let i = block + j
                    let k = i + half
                    let a = data[i]
                    let b = data[k]
                    // Butterfly: (a, b) -> (a + b, b * beta)
                    data[i] = a + b
                    data[k] = b * beta
                }
                block += blockSize
            }
        }
    }

    /// Inverse additive FFT (in-place)
    /// Transforms evaluations back to coefficients
    public static func inverse(data: inout [BinaryField32], logN: Int) {
        let n = 1 << logN
        precondition(data.count == n)

        let basis = subspaceBasis(logN: logN)

        // Reverse butterfly: logN stages in reverse order
        for l in stride(from: logN - 1, through: 0, by: -1) {
            let blockSize = n >> l
            let half = blockSize >> 1
            let beta = basis[logN - 1 - l]
            let betaInv = beta.inverse()

            var block = 0
            while block < n {
                for j in 0..<half {
                    let i = block + j
                    let k = i + half
                    let a = data[i]
                    let b = data[k]
                    // Inverse butterfly: (a, b) -> (a + b*betaInv, b*betaInv)
                    // Wait — inverse of (a,b)->(a+b, b*beta) is:
                    // From a'=a+b, b'=b*beta:  b=b'/beta, a=a'+b=a'+b'/beta
                    // So: (a', b') -> (a' + b'/beta, b'/beta)
                    let bOrig = b * betaInv
                    data[i] = a + bOrig
                    data[k] = bOrig
                }
                block += blockSize
            }
        }
    }

    /// Verify FFT correctness: forward then inverse should be identity
    public static func verifyRoundtrip(logN: Int) -> Bool {
        let n = 1 << logN
        var rng: UInt64 = 0xDEAD_BEEF
        var original = [BinaryField32]()
        original.reserveCapacity(n)
        for _ in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1
            original.append(BinaryField32(value: UInt32(truncatingIfNeeded: rng)))
        }
        var data = original
        forward(data: &data, logN: logN)
        inverse(data: &data, logN: logN)
        return data == original
    }
}

// MARK: - GPU-Accelerated Binary FFT Engine

public class BinaryFFTEngine {
    public static let version = "1.0.0"

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    let batchAddFunction: MTLComputePipelineState
    let batchMulFunction: MTLComputePipelineState
    let butterflyFunction: MTLComputePipelineState

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try BinaryFFTEngine.compileShaders(device: device)

        guard let addFn = library.makeFunction(name: "bt_batch_add"),
              let mulFn = library.makeFunction(name: "bt_batch_mul"),
              let bflyFn = library.makeFunction(name: "bt_additive_butterfly") else {
            throw MSMError.missingKernel
        }

        self.batchAddFunction = try device.makeComputePipelineState(function: addFn)
        self.batchMulFunction = try device.makeComputePipelineState(function: mulFn)
        self.butterflyFunction = try device.makeComputePipelineState(function: bflyFn)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let source = try String(contentsOfFile: shaderDir + "/fields/binary_tower.metal", encoding: .utf8)

        let clean = source
            .replacingOccurrences(of: "#ifndef BINARY_TOWER_METAL", with: "")
            .replacingOccurrences(of: "#define BINARY_TOWER_METAL", with: "")
            .replacingOccurrences(of: "#endif // BINARY_TOWER_METAL", with: "")

        let options = MTLCompileOptions()
        options.fastMathEnabled = true
        return try device.makeLibrary(source: clean, options: options)
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let path = url.appendingPathComponent("fields/binary_tower.metal").path
                if FileManager.default.fileExists(atPath: path) {
                    return url.path
                }
            }
        }
        let candidates = [
            "\(execDir)/../Sources/Shaders",
            "./Sources/Shaders",
            "\(execDir)/Sources/Shaders",
            "\(execDir)/../../Sources/Shaders",
        ]
        for c in candidates {
            if FileManager.default.fileExists(atPath: c + "/fields/binary_tower.metal") {
                return c
            }
        }
        return "./Sources/Shaders"
    }

    /// GPU batch add: out[i] = a[i] XOR b[i] — should be bandwidth-bound
    public func batchAdd(a: [UInt32], b: [UInt32]) throws -> [UInt32] {
        let n = a.count
        precondition(b.count == n)

        let bufA = device.makeBuffer(bytes: a, length: n * 4, options: .storageModeShared)!
        let bufB = device.makeBuffer(bytes: b, length: n * 4, options: .storageModeShared)!
        let bufOut = device.makeBuffer(length: n * 4, options: .storageModeShared)!

        let cb = commandQueue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchAddFunction)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufOut, offset: 0, index: 2)
        let tgSize = min(256, batchAddFunction.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        var result = [UInt32](repeating: 0, count: n)
        memcpy(&result, bufOut.contents(), n * 4)
        return result
    }

    /// GPU batch multiply: out[i] = a[i] * b[i] in GF(2^32)
    public func batchMul(a: [UInt32], b: [UInt32]) throws -> [UInt32] {
        let n = a.count
        precondition(b.count == n)

        let bufA = device.makeBuffer(bytes: a, length: n * 4, options: .storageModeShared)!
        let bufB = device.makeBuffer(bytes: b, length: n * 4, options: .storageModeShared)!
        let bufOut = device.makeBuffer(length: n * 4, options: .storageModeShared)!

        let cb = commandQueue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(batchMulFunction)
        enc.setBuffer(bufA, offset: 0, index: 0)
        enc.setBuffer(bufB, offset: 0, index: 1)
        enc.setBuffer(bufOut, offset: 0, index: 2)
        let tgSize = min(256, batchMulFunction.maxTotalThreadsPerThreadgroup)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()

        var result = [UInt32](repeating: 0, count: n)
        memcpy(&result, bufOut.contents(), n * 4)
        return result
    }

    /// GPU additive FFT over GF(2^32)
    /// Uses butterfly kernel for each stage
    public func forwardFFT(data: inout [UInt32], logN: Int) throws {
        let n = 1 << logN
        precondition(data.count == n)

        let basis = BinaryFFT.subspaceBasis(logN: logN)
        let bufData = device.makeBuffer(bytes: data, length: n * 4, options: .storageModeShared)!

        let cb = commandQueue.makeCommandBuffer()!

        for l in 0..<logN {
            let blockSize = n >> l
            let half = blockSize >> 1
            let beta = basis[logN - 1 - l]

            // Create twiddle buffer (all same value for this simple basis)
            var twArr = [UInt32](repeating: beta.toUInt32, count: half)
            let bufTw = device.makeBuffer(bytes: &twArr, length: half * 4, options: .storageModeShared)!

            var halfN = UInt32(half)
            var stride = UInt32(blockSize)

            let enc = cb.makeComputeCommandEncoder()!
            enc.setComputePipelineState(butterflyFunction)
            enc.setBuffer(bufData, offset: 0, index: 0)
            enc.setBuffer(bufTw, offset: 0, index: 1)
            enc.setBytes(&halfN, length: 4, index: 2)
            enc.setBytes(&stride, length: 4, index: 3)
            let numThreads = n / 2
            let tgSize = min(256, butterflyFunction.maxTotalThreadsPerThreadgroup)
            enc.dispatchThreads(MTLSize(width: numThreads, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
            enc.endEncoding()
        }

        cb.commit()
        cb.waitUntilCompleted()

        memcpy(&data, bufData.contents(), n * 4)
    }
}
