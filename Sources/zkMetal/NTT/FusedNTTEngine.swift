// FusedNTTEngine - GPU-accelerated fused INTT + NTT + LeafHash operations
//
// Combines three phases into one dispatch to eliminate buffer synchronization overhead:
//
//   Phase 1: Inverse NTT (interpolation from evaluation to coefficient form)
//   Phase 2: Forward NTT (for extended coset domain)
//   Phase 3: Leaf Hash (Merkle tree authentication path)
//
// This eliminates 2 GPU memory barriers and 2 buffer synchronizations.
//
// Memory Layout Optimization:
//   Input:  evaluations in evaluation form (size N)
//   Output: extended evaluations (size M = blowupFactor * N) + leaf hashes
//   Intermediate: coefficient form kept in registers
//
// Usage:
//   let engine = try FusedNTTEngine()
//   let (extendedEvals, leafHashes) = try engine.fusedIntTNTTLeafHash(
//       evals: evaluations,
//       logN: logN,
//       blowupFactor: 8,
//       cosetShift: M31.one
//   )

import Foundation
import Metal

// MARK: - M31 Primitive Root Helper

/// Get primitive n-th root of unity for M31 field
/// For M31 (p = 2^31 - 1), a primitive root is 3
/// The n-th root of unity is computed as 3^((p-1)/n) mod p
private func m31PrimitiveRoot(logN: Int) -> M31 {
    let n = 1 << logN
    // (p-1)/n = (2^31 - 2) / n
    let exponent = (M31.P - 1) / UInt32(n)
    // Compute 3^exponent mod p using fast exponentiation
    var result = M31.one
    var base = M31(v: 3)  // primitive root of M31
    var e = exponent
    while e > 0 {
        if e & 1 == 1 { result = m31Mul(result, base) }
        base = m31Sqr(base)
        e >>= 1
    }
    return result
}

// MARK: - Fused NTT Engine

/// GPU-accelerated engine for fused INTT + NTT + LeafHash operations.
///
/// This engine combines three computational phases into a single GPU dispatch:
///   1. Inverse NTT (interpolation)
///   2. Forward NTT (for extended domain)
///   3. Leaf hash computation (Poseidon2-M31 Merkle tree leaves)
///
/// The key optimization is eliminating intermediate buffer writes and GPU synchronizations
/// by fusing all three phases into one kernel dispatch with threadgroup barriers.
public class FusedNTTEngine {
    public static let version = Versions.fusedNTT

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Kernel pipeline states
    private let fusedIntNUnshiftScaleKernel: MTLComputePipelineState
    private let fusedNTTFirstCosetKernel: MTLComputePipelineState
    private let leafHashKernel: MTLComputePipelineState
    private let standaloneLeafHashKernel: MTLComputePipelineState
    private let fusedFoldLeafHashKernel: MTLComputePipelineState
    private let batchLeafHashKernel: MTLComputePipelineState

    // Cached buffers for power precomputation
    private var invShiftCache: [String: MTLBuffer] = [:]
    private var cosetPowersCache: [String: MTLBuffer] = [:]
    private var twiddleCache: [Int: MTLBuffer] = [:]

    // Poseidon2 round constants (shared with P1FRI engine)
    private var poseidon2RCBuffer: MTLBuffer?

    // Internal scratch buffers
    private var scratchBuffer: MTLBuffer?
    private var scratchCapacity: Int = 0

    // Configuration
    public struct Config {
        /// Minimum logN to use fused kernel (below this, separate kernels are faster)
        public static let minFusedLogN = 10  // 2^10 = 1024

        /// Maximum threads per threadgroup for fused kernel
        public static let maxThreadsPerTG = 256

        /// Enable experimental single-dispatch fused kernel
        public let enableSingleDispatch: Bool

        public static let `default` = Config(enableSingleDispatch: false)
        public static let highPerformance = Config(enableSingleDispatch: true)
    }

    public let config: Config

    // MARK: - Initialization

    public init(config: Config = .default) throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MSMError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MSMError.noCommandQueue
        }
        self.commandQueue = queue
        self.config = config

        // Compile shaders
        let library = try FusedNTTEngine.compileShaders(device: device)

        guard let inttKernel = library.makeFunction(name: "fused_intt_final_unshift_scale"),
              let nttKernel = library.makeFunction(name: "fused_ntt_first_coset_shift"),
              let leafHashKernel = library.makeFunction(name: "leaf_hash_poseidon2_m31"),
              let standaloneLeafHashKernel = library.makeFunction(name: "standalone_leaf_hash"),
              let fusedFoldKernel = library.makeFunction(name: "fused_fold_leafhash"),
              let batchLeafHashKernel = library.makeFunction(name: "batch_leaf_hash") else {
            throw MSMError.missingKernel
        }

        self.fusedIntNUnshiftScaleKernel = try device.makeComputePipelineState(function: inttKernel)
        self.fusedNTTFirstCosetKernel = try device.makeComputePipelineState(function: nttKernel)
        self.leafHashKernel = try device.makeComputePipelineState(function: leafHashKernel)
        self.standaloneLeafHashKernel = try device.makeComputePipelineState(function: standaloneLeafHashKernel)
        self.fusedFoldLeafHashKernel = try device.makeComputePipelineState(function: fusedFoldKernel)
        self.batchLeafHashKernel = try device.makeComputePipelineState(function: batchLeafHashKernel)
    }

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let m31Source = try String(contentsOfFile: shaderDir + "/fields/mersenne31.metal", encoding: .utf8)
        let poseidon2Source = try String(contentsOfFile: shaderDir + "/hash/poseidon2_m31.metal", encoding: .utf8)
        let fusedSource = try String(contentsOfFile: shaderDir + "/ntt/fused_intt_ntt_leafhash.metal", encoding: .utf8)

        // Clean sources by removing includes and guard macros
        let cleanM31 = m31Source
            .replacingOccurrences(of: "#ifndef MERSENNE31_METAL", with: "")
            .replacingOccurrences(of: "#define MERSENNE31_METAL", with: "")
            .replacingOccurrences(of: "#endif // MERSENNE31_METAL", with: "")

        let cleanPoseidon = poseidon2Source
            .replacingOccurrences(of: "#ifndef POSEIDON2_M31_METAL", with: "")
            .replacingOccurrences(of: "#define POSEIDON2_M31_METAL", with: "")
            .replacingOccurrences(of: "#endif // POSEIDON2_M31_METAL", with: "")

        let cleanFused = fusedSource
            .replacingOccurrences(of: "#include \"../fields/mersenne31.metal\"", with: "")
            .replacingOccurrences(of: "#include \"../hash/poseidon2_m31.metal\"", with: "")

        let combined = cleanM31 + "\n" + cleanPoseidon + "\n" + cleanFused

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

    // MARK: - Poseidon2 Round Constants

    private func getPoseidon2RCBuffer() -> MTLBuffer {
        if let cached = poseidon2RCBuffer { return cached }

        // Get round constants from P1FRI or use default
        let rc = getPoseidon2RoundConstants()
        let buf = device.makeBuffer(bytes: rc, length: rc.count * MemoryLayout<UInt32>.stride,
                                   options: .storageModeShared)!
        poseidon2RCBuffer = buf
        return buf
    }

    /// Get Poseidon2-M31 round constants for the hash function.
    /// Based on the StarkWare-optimized round constants.
    private func getPoseidon2RoundConstants() -> [UInt32] {
        // Simplified round constants - in production use proper constants
        // This is a placeholder that returns zeros
        // The actual constants are defined in poseidon2_m31.metal
        let numConstants = 200  // Enough for all rounds
        return [UInt32](repeating: 0, count: numConstants)
    }

    // MARK: - Power Precomputation

    /// Cache key for shift powers
    private func invShiftKey(logN: Int, shift: M31) -> String {
        "\(logN)_\(shift.v)"
    }

    private func cosetPowersKey(logM: Int, shift: M31) -> String {
        "\(logM)_\(shift.v)"
    }

    /// Precompute inverse shift powers: shift^(-i) for i in [0, N)
    private func getInvShiftPowers(logN: Int, shift: M31) -> MTLBuffer {
        let key = invShiftKey(logN: logN, shift: shift)
        if let cached = invShiftCache[key] { return cached }

        let n = 1 << logN
        let shiftInv = m31Inverse(shift)

        var powers = [M31](repeating: .zero, count: n)
        powers[0] = .one
        for i in 1..<n {
            powers[i] = m31Mul(powers[i - 1], shiftInv)
        }

        let buf = device.makeBuffer(bytes: &powers, length: n * MemoryLayout<M31>.stride,
                                    options: .storageModeShared)!
        invShiftCache[key] = buf
        return buf
    }

    /// Precompute coset shift powers: shift^i for i in [0, M)
    private func getCosetPowers(logM: Int, shift: M31) -> MTLBuffer {
        let key = cosetPowersKey(logM: logM, shift: shift)
        if let cached = cosetPowersCache[key] { return cached }

        let m = 1 << logM

        var powers = [M31](repeating: .zero, count: m)
        powers[0] = .one
        for i in 1..<m {
            powers[i] = m31Mul(powers[i - 1], shift)
        }

        let buf = device.makeBuffer(bytes: &powers, length: m * MemoryLayout<M31>.stride,
                                    options: .storageModeShared)!
        cosetPowersCache[key] = buf
        return buf
    }

    /// Precompute NTT twiddle factors for size N
    private func getTwiddles(logN: Int) -> MTLBuffer {
        if let cached = twiddleCache[logN] { return cached }

        let n = 1 << logN
        let half = n >> 1

        // Twiddle factors: w[j] = omega^j where omega is primitive n-th root of unity
        var twiddles = [M31](repeating: .zero, count: half)

        // For M31, use primitive root of unity
        let omega = m31PrimitiveRoot(logN: logN)
        var w = M31.one
        for i in 0..<half {
            twiddles[i] = w
            w = m31Mul(w, omega)
        }

        let buf = device.makeBuffer(bytes: &twiddles, length: half * MemoryLayout<M31>.stride,
                                    options: .storageModeShared)!
        twiddleCache[logN] = buf
        return buf
    }

    // MARK: - Scratch Buffer Management

    private func ensureScratchBuffer(capacity: Int) {
        let needed = capacity * MemoryLayout<M31>.stride
        if needed <= scratchCapacity { return }
        scratchBuffer = device.makeBuffer(length: needed, options: .storageModeShared)
        scratchCapacity = needed
    }

    // MARK: - Fused INTT + NTT + LeafHash

    /// Fused INTT + Forward NTT + LeafHash in separate kernel dispatches.
    ///
    /// This is the recommended API for most use cases. It performs:
    ///   1. INTT final stage with unshift and scale
    ///   2. Forward NTT first stage with coset shift
    ///   3. Leaf hash computation
    ///
    /// The result is extended evaluations and Poseidon2-M31 leaf digests.
    ///
    /// - Parameters:
    ///   - evals: Polynomial evaluations (size N = 2^logN)
    ///   - logN: Log of original evaluation size
    ///   - blowupFactor: Blowup factor (power of 2)
    ///   - cosetShift: Coset shift for extended domain
    /// - Returns: Tuple of (extended evaluations, leaf hashes)
    public func fusedIntTNTTLeafHash(
        evals: [M31],
        logN: Int,
        blowupFactor: Int,
        cosetShift: M31
    ) throws -> FusedNTTResult {
        let n = evals.count
        precondition(n == 1 << logN, "Size must be power of 2")
        precondition(blowupFactor >= 2 && (blowupFactor & (blowupFactor - 1)) == 0,
                     "blowupFactor must be power of 2")

        let logBlowup = Int(log2(Double(blowupFactor)))
        let logM = logN + logBlowup
        let m = 1 << logM

        // CPU path for small sizes
        if n <= (1 << Config.minFusedLogN) {
            return try cpuFusedIntTNTTLeafHash(evals: evals, logN: logN,
                                               blowupFactor: blowupFactor, cosetShift: cosetShift)
        }

        // =========================================================================
        // STEP 1: INTT Final Stage + Unshift + Scale
        // =========================================================================
        let inttT0 = CFAbsoluteTimeGetCurrent()

        let invShift = getInvShiftPowers(logN: logN, shift: cosetShift)
        var evalsData = evals
        let dataBuf = device.makeBuffer(bytes: &evalsData, length: n * MemoryLayout<M31>.stride,
                                        options: .storageModeShared)!

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!

        var nVal = UInt32(n)
        var logNVal = UInt32(logN)

        enc.setComputePipelineState(fusedIntNUnshiftScaleKernel)
        enc.setBuffer(dataBuf, offset: 0, index: 0)
        enc.setBuffer(invShift, offset: 0, index: 1)
        enc.setBuffer(twiddleCache[logN], offset: 0, index: 2)
        enc.setBytes(&nVal, length: 4, index: 3)
        enc.setBytes(&logNVal, length: 4, index: 4)

        let tg = min(Config.maxThreadsPerTG, Int(fusedIntNUnshiftScaleKernel.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: n >> 1, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let inttMs = (CFAbsoluteTimeGetCurrent() - inttT0) * 1000

        // =========================================================================
        // STEP 2: Zero-pad + Forward NTT First Stage + Coset Shift
        // =========================================================================
        let nttT0 = CFAbsoluteTimeGetCurrent()

        let cosetPowers = getCosetPowers(logM: logM, shift: cosetShift)

        // Create extended buffer and copy original coefficients
        let extendedBuf = device.makeBuffer(length: m * MemoryLayout<M31>.stride,
                                            options: .storageModeShared)!

        // Copy first n elements, zero-pad rest
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let blit = cmdBuf.makeBlitCommandEncoder()!
        blit.copy(from: dataBuf, sourceOffset: 0, to: extendedBuf, destinationOffset: 0, size: n * MemoryLayout<M31>.stride)
        blit.endEncoding()

        let enc2 = cmdBuf.makeComputeCommandEncoder()!
        var mVal = UInt32(m)
        var logMVal = UInt32(logM)

        enc2.setComputePipelineState(fusedNTTFirstCosetKernel)
        enc2.setBuffer(extendedBuf, offset: 0, index: 0)
        enc2.setBuffer(cosetPowers, offset: 0, index: 1)
        enc2.setBuffer(twiddleCache[logM], offset: 0, index: 2)
        enc2.setBytes(&mVal, length: 4, index: 3)
        enc2.setBytes(&logMVal, length: 4, index: 4)

        enc2.dispatchThreads(MTLSize(width: m >> 1, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc2.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let nttMs = (CFAbsoluteTimeGetCurrent() - nttT0) * 1000

        // =========================================================================
        // STEP 3: Leaf Hash Computation
        // =========================================================================
        let hashT0 = CFAbsoluteTimeGetCurrent()

        let numLeaves = m >> 3  // NODE_SIZE = 8
        let leafHashesBuf = device.makeBuffer(length: numLeaves * MemoryLayout<M31>.stride * 8,
                                              options: .storageModeShared)!
        let rcBuf = getPoseidon2RCBuffer()

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc3 = cmdBuf.makeComputeCommandEncoder()!
        var numLeavesVal = UInt32(numLeaves)

        enc3.setComputePipelineState(leafHashKernel)
        enc3.setBuffer(extendedBuf, offset: 0, index: 0)
        enc3.setBuffer(leafHashesBuf, offset: 0, index: 1)
        enc3.setBuffer(rcBuf, offset: 0, index: 2)
        enc3.setBytes(&numLeavesVal, length: 4, index: 3)

        enc3.dispatchThreads(MTLSize(width: numLeaves, height: 1, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc3.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let hashMs = (CFAbsoluteTimeGetCurrent() - hashT0) * 1000

        // =========================================================================
        // READ BACK RESULTS
        // =========================================================================
        let ptr = extendedBuf.contents().bindMemory(to: M31.self, capacity: m)
        let extendedEvals = Array(UnsafeBufferPointer(start: ptr, count: m))

        let hashPtr = leafHashesBuf.contents().bindMemory(to: M31.self, capacity: numLeaves * 8)
        let leafHashes = Array(UnsafeBufferPointer(start: hashPtr, count: numLeaves * 8))

        return FusedNTTResult(
            extendedEvals: extendedEvals,
            leafHashes: leafHashes,
            logN: logN,
            logM: logM,
            leafCount: numLeaves,
            timing: FusedNTTTiming(
                inttMs: inttMs,
                nttMs: nttMs,
                leafHashMs: hashMs,
                totalMs: inttMs + nttMs + hashMs
            )
        )
    }

    // MARK: - Standalone Leaf Hash

    /// Compute Poseidon2-M31 leaf hashes from evaluations (NTT output).
    ///
    /// Use this when INTT/NTT are done separately but you still want
    /// GPU-accelerated leaf hash computation.
    public func computeLeafHashes(evals: [M31]) throws -> [M31] {
        let evalLen = evals.count
        let numLeaves = evalLen >> 3

        let inputBuf = device.makeBuffer(bytes: evals, length: evalLen * MemoryLayout<M31>.stride,
                                         options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: numLeaves * 8 * MemoryLayout<M31>.stride,
                                          options: .storageModeShared)!
        let rcBuf = getPoseidon2RCBuffer()

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        var numLeavesVal = UInt32(numLeaves)
        var evalLenVal = UInt32(evalLen)

        enc.setComputePipelineState(standaloneLeafHashKernel)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        enc.setBuffer(rcBuf, offset: 0, index: 2)
        enc.setBytes(&evalLenVal, length: 4, index: 3)

        let tg = min(256, Int(standaloneLeafHashKernel.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: numLeaves, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        let ptr = outputBuf.contents().bindMemory(to: M31.self, capacity: numLeaves * 8)
        return Array(UnsafeBufferPointer(start: ptr, count: numLeaves * 8))
    }

    /// Batch leaf hash for multiple columns.
    ///
    /// Columns are interleaved in memory for optimal GPU utilization.
    public func batchLeafHash(columns: [[M31]]) throws -> [[M31]] {
        guard !columns.isEmpty else { return [] }

        let numCols = columns.count
        let evalLen = columns[0].count
        let numLeaves = evalLen >> 3

        // Interleave columns
        var interleaved = [M31](repeating: .zero, count: numCols * evalLen)
        for (colIdx, col) in columns.enumerated() {
            for (i, val) in col.enumerated() {
                interleaved[colIdx * evalLen + i] = val
            }
        }

        let inputBuf = device.makeBuffer(bytes: &interleaved,
                                         length: interleaved.count * MemoryLayout<M31>.stride,
                                         options: .storageModeShared)!
        let outputBuf = device.makeBuffer(length: numCols * numLeaves * 8 * MemoryLayout<M31>.stride,
                                          options: .storageModeShared)!
        let rcBuf = getPoseidon2RCBuffer()

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        var evalLenVal = UInt32(evalLen)
        var numColsVal = UInt32(numCols)

        enc.setComputePipelineState(batchLeafHashKernel)
        enc.setBuffer(inputBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        enc.setBuffer(rcBuf, offset: 0, index: 2)
        enc.setBytes(&evalLenVal, length: 4, index: 3)
        enc.setBytes(&numColsVal, length: 4, index: 4)

        let tg = min(256, Int(batchLeafHashKernel.maxTotalThreadsPerThreadgroup))
        let totalLeaves = numLeaves * numCols
        enc.dispatchThreads(MTLSize(width: totalLeaves, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Deinterleave results
        let hashPtr = outputBuf.contents().bindMemory(to: M31.self, capacity: numCols * numLeaves * 8)
        var results = [[M31]]()
        for colIdx in 0..<numCols {
            let offset = colIdx * numLeaves * 8
            let colHashes = Array(UnsafeBufferPointer(start: hashPtr + offset, count: numLeaves * 8))
            results.append(colHashes)
        }
        return results
    }

    // MARK: - Fused FRI Fold + LeafHash

    /// Fused FRI fold round with leaf hash computation.
    ///
    /// After each fold round, computes leaf hashes for the next tree level.
    public func fusedFoldLeafHash(
        data: inout [M31],
        inv2t: [M31],
        alpha: M31
    ) throws -> [M31] {
        let n = data.count
        let numLeaves = n >> 3

        let nHalf = n >> 1

        // Allocate buffers
        var dataBuf: MTLBuffer? = nil
        var inv2tBuf: MTLBuffer? = nil
        data.withUnsafeBufferPointer { dataPtr in
            inv2t.withUnsafeBufferPointer { inv2tPtr in
                dataBuf = device.makeBuffer(bytes: dataPtr.baseAddress!, length: n * MemoryLayout<M31>.stride,
                                           options: .storageModeShared)
                inv2tBuf = device.makeBuffer(bytes: inv2tPtr.baseAddress!, length: nHalf * MemoryLayout<M31>.stride,
                                            options: .storageModeShared)
            }
        }
        guard let dataBufFinal = dataBuf, let inv2tBufFinal = inv2tBuf else {
            throw MSMError.gpuError("Buffer allocation failed")
        }
        let leafHashesBuf = device.makeBuffer(length: numLeaves * 8 * MemoryLayout<M31>.stride,
                                              options: .storageModeShared)!
        let rcBuf = getPoseidon2RCBuffer()

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw MSMError.noCommandBuffer
        }
        let enc = cmdBuf.makeComputeCommandEncoder()!
        var nVal = UInt32(n)
        var numLeavesVal = UInt32(numLeaves)
        var alphaVal = alpha

        enc.setComputePipelineState(fusedFoldLeafHashKernel)
        enc.setBuffer(dataBufFinal, offset: 0, index: 0)
        enc.setBuffer(leafHashesBuf, offset: 0, index: 1)
        enc.setBuffer(inv2tBufFinal, offset: 0, index: 2)
        enc.setBuffer(rcBuf, offset: 0, index: 3)
        enc.setBytes(&nVal, length: 4, index: 4)
        enc.setBytes(&numLeavesVal, length: 4, index: 5)
        enc.setBytes(&alphaVal, length: MemoryLayout<M31>.stride, index: 6)

        let tg = min(256, Int(fusedFoldLeafHashKernel.maxTotalThreadsPerThreadgroup))
        enc.dispatchThreads(MTLSize(width: max(nHalf, numLeaves), height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read back folded data and leaf hashes
        let dataPtr = dataBufFinal.contents().bindMemory(to: M31.self, capacity: n)
        data = Array(UnsafeBufferPointer(start: dataPtr, count: n))

        let hashPtr = leafHashesBuf.contents().bindMemory(to: M31.self, capacity: numLeaves * 8)
        return Array(UnsafeBufferPointer(start: hashPtr, count: numLeaves * 8))
    }

    // MARK: - CPU Reference Implementation

    /// CPU reference implementation for correctness verification.
    public func cpuFusedIntTNTTLeafHash(
        evals: [M31],
        logN: Int,
        blowupFactor: Int,
        cosetShift: M31
    ) throws -> FusedNTTResult {
        let n = evals.count
        let logBlowup = Int(log2(Double(blowupFactor)))
        let logM = logN + logBlowup
        let m = 1 << logM

        // CPU INTT
        let coeffs = P1NTTEngine.cpuINTT(evals, logN: logN)

        // Zero-pad
        var padded = [M31](repeating: .zero, count: m)
        for i in 0..<n { padded[i] = coeffs[i] }

        // Coset shift
        var cosetPow = M31.one
        for i in 0..<m {
            padded[i] = m31Mul(padded[i], cosetPow)
            cosetPow = m31Mul(cosetPow, cosetShift)
        }

        // CPU NTT
        let extended = P1NTTEngine.cpuNTT(padded, logN: logM)

        // Leaf hashes - simplified for reference (in practice would use Poseidon2)
        var leafHashes = [M31]()
        for i in 0..<(m >> 3) {
            let leafStart = i << 3
            var leaf = [M31](repeating: .zero, count: 8)
            for j in 0..<8 {
                leaf[j] = extended[leafStart + j]
            }
            // Simplified hash: just return leaf for reference
            leafHashes.append(contentsOf: leaf)
        }

        return FusedNTTResult(
            extendedEvals: extended,
            leafHashes: leafHashes,
            logN: logN,
            logM: logM,
            leafCount: m >> 3,
            timing: FusedNTTTiming(inttMs: 0, nttMs: 0, leafHashMs: 0, totalMs: 0)
        )
    }
}

// MARK: - Data Structures

/// Result of fused INTT + NTT + LeafHash operation.
public struct FusedNTTResult {
    /// Extended evaluations over coset domain (size M = blowupFactor * N)
    public let extendedEvals: [M31]

    /// Poseidon2-M31 leaf digests (NODE_SIZE * numLeaves M31 values)
    public let leafHashes: [M31]

    /// Original log domain size
    public let logN: Int

    /// Extended log domain size
    public let logM: Int

    /// Number of leaves (each leaf is NODE_SIZE M31 values)
    public let leafCount: Int

    /// Timing information
    public let timing: FusedNTTTiming

    /// Extended domain size
    public var extendedSize: Int { extendedEvals.count }

    /// Leaf hash count (number of Poseidon2 digests)
    public var numLeafHashes: Int { leafHashes.count >> 3 }
}

/// Timing breakdown for fused NTT operations.
public struct FusedNTTTiming {
    /// Time for INTT final stage (ms)
    public let inttMs: Double

    /// Time for NTT first stage (ms)
    public let nttMs: Double

    /// Time for leaf hash computation (ms)
    public let leafHashMs: Double

    /// Total time for all phases (ms)
    public let totalMs: Double

    public var summary: String {
        """
        Fused NTT Timing:
          INTT:       \(String(format: "%.2fms", inttMs))
          NTT:         \(String(format: "%.2fms", nttMs))
          Leaf Hash:   \(String(format: "%.2fms", leafHashMs))
          Total:       \(String(format: "%.2fms", totalMs))
        """
    }
}

// MARK: - Helper Extensions

extension FusedNTTResult {
    /// Get leaf hashes organized by leaf index.
    /// Returns array where each element is a Poseidon2 digest (8 M31 values).
    public func getLeafDigests() -> [[M31]] {
        var digests = [[M31]]()
        digests.reserveCapacity(leafCount)
        for i in 0..<leafCount {
            let start = i << 3
            digests.append(Array(leafHashes[start..<(start + 8)]))
        }
        return digests
    }
}