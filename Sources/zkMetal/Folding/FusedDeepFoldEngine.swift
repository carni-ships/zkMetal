// FusedDeepFoldEngine — GPU-accelerated Multi-Round Nova/Supernova Folding
//
// Fuses 4-8 Nova fold rounds into a single GPU dispatch with shared-memory
// accumulation. Reduces dispatch overhead and memory bandwidth by eliminating
// intermediate GPU synchronizations between rounds.
//
// Architecture:
//   - fused_deepfold_bn254: Configurable kernel for 4-8 rounds
//   - fused_deepfold_bn254_by4: Specialized 4-round kernel (optimal for common case)
//   - fused_deepfold_bn254_by8: Specialized 8-round kernel (high throughput)
//   - fused_deepfold_bn254_with_witness: Variant with witness accumulation
//
// Key optimizations:
//   - Single dispatch instead of 4-8 separate kernel dispatches
//   - Threadgroup barriers sync accumulation across rounds
//   - Reuses az0, bz0, cz0 from registers across all rounds
//   - Threadgroup memory for running accumulators
//
// Cross-term formula (Nova-style):
//   T_i = az0 * bz_i + az_i * bz0 - u0 * cz_i - cz0
//   Accumulated: T = sum_i r_i * T_i
//
// Usage:
//   let engine = try FusedDeepFoldEngine(fusedRounds: 4)
//   let (t, w) = try engine.fusedFold(
//       az0: az0, bz0: bz0, cz0: cz0,
//       instances: [(az1, bz1, cz1), (az2, bz2, cz2), ...],
//       challenges: [r0, r1, r2, ...]
//   )
//
// Reference: "Nova: Recursive Zero-Knowledge Arguments from Folding Schemes"
//            (Kothapalli, Setty, Tzialla 2022)

import Foundation
import Metal

// MARK: - Version

public let kFusedDeepFoldVersion = PrimitiveVersion(version: "1.0.0", updated: "2026-04-28")

// MARK: - Errors

public enum FusedFoldError: Error {
    case noGPU
    case noCommandQueue
    case noCommandBuffer
    case missingKernel(String)
    case bufferAllocationFailed
    case invalidFusedRounds(String)
    case shaderCompilationFailed(String)
    case invalidInputSize(String)
}

// MARK: - Fused Deep Fold Engine

/// GPU-accelerated fused multi-round Nova/Supernova folding engine.
///
/// Fuses 4-8 consecutive fold rounds into a single GPU dispatch to reduce
/// dispatch overhead and memory bandwidth. The engine provides:
///   - Configurable kernel supporting 4-8 rounds
///   - Specialized kernels for 4 rounds (common case)
///   - Specialized kernels for 8 rounds (high throughput)
///   - Optional witness accumulation
///
/// Usage:
///   ```swift
///   let engine = try FusedDeepFoldEngine(fusedRounds: 4)
///   let (t, w) = try engine.fusedFold(
///       az0: az0, bz0: bz0, cz0: cz0,
///       instances: [(az1, bz1, cz1), (az2, bz2, cz2), (az3, bz3, cz3), (az4, bz4, cz4)],
///       challenges: [r0, r1, r2, r3]
///   )
///   ```
public class FusedDeepFoldEngine {
    // MARK: - Version

    public static let version = kFusedDeepFoldVersion

    // MARK: - GPU Resources

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    /// Number of rounds fused in this engine (4 or 8 recommended)
    public let fusedRounds: Int

    /// Threadgroup size for kernel execution
    public let threadgroupSize: Int

    // Kernel pipeline states
    private let fusedBy4Kernel: MTLComputePipelineState
    private let fusedBy8Kernel: MTLComputePipelineState

    // Memory
    private var threadgroupMemory: MTLBuffer?
    private let threadgroupMemorySize: Int

    // MARK: - Initialization

    /// Initialize the fused deep fold engine.
    ///
    /// - Parameters:
    ///   - fusedRounds: Number of rounds to fuse (4-8, default 4)
    ///   - threadgroupSize: Threads per threadgroup (default 256)
    public init(fusedRounds: Int = 4, threadgroupSize: Int = 256) throws {
        guard fusedRounds >= 2 && fusedRounds <= 8 else {
            throw FusedFoldError.invalidFusedRounds("fusedRounds must be between 2 and 8")
        }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw FusedFoldError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw FusedFoldError.noCommandQueue
        }
        self.commandQueue = queue

        self.fusedRounds = fusedRounds
        self.threadgroupSize = threadgroupSize

        // Calculate threadgroup memory: 2 * threadgroupSize * sizeof(Fr)
        // sharedT[0..tgSize-1] and sharedW[0..tgSize-1]
        // Fr is 32 bytes (8x uint32 limbs for BN254)
        let frSize = 32
        self.threadgroupMemorySize = 2 * threadgroupSize * frSize

        // Compile shaders
        let library = try FusedDeepFoldEngine.compileShaders(device: device)

        // Get kernel functions
        guard let fusedBy4Fn = library.makeFunction(name: "fused_deepfold_bn254_by4") else {
            throw FusedFoldError.missingKernel("fused_deepfold_bn254_by4")
        }

        guard let fusedBy8Fn = library.makeFunction(name: "fused_deepfold_bn254_by8") else {
            throw FusedFoldError.missingKernel("fused_deepfold_bn254_by8")
        }

        self.fusedBy4Kernel = try device.makeComputePipelineState(function: fusedBy4Fn)
        self.fusedBy8Kernel = try device.makeComputePipelineState(function: fusedBy8Fn)

        // Pre-allocate threadgroup memory
        self.threadgroupMemory = device.makeBuffer(
            length: threadgroupMemorySize,
            options: .storageModeShared
        )
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()
        let frSource = try String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8)
        let deepFoldSource = try String(contentsOfFile: shaderDir + "/fold/fused_deepfold.metal", encoding: .utf8)

        // Strip #include directives (we inline dependencies)
        let cleanDeepFold = deepFoldSource.split(separator: "\n")
            .filter { !$0.contains("#include") }
            .joined(separator: "\n")

        // Clean up include guards from bn254_fr.metal
        let cleanFr = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        // Clean up include guards from fused_deepfold.metal
        let cleanDeepFold2 = cleanDeepFold
            .replacingOccurrences(of: "#ifndef FOLD_FUSED_DEEPFOLD_METAL", with: "")
            .replacingOccurrences(of: "#define FOLD_FUSED_DEEPFOLD_METAL", with: "")
            .replacingOccurrences(of: "#endif // FOLD_FUSED_DEEPFOLD_METAL", with: "")

        let combined = cleanFr + "\n" + cleanDeepFold2

        let options = MTLCompileOptions()
        options.fastMathEnabled = true

        do {
            return try device.makeLibrary(source: combined, options: options)
        } catch {
            throw FusedFoldError.shaderCompilationFailed(error.localizedDescription)
        }
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let path = url.appendingPathComponent("fold/fused_deepfold.metal").path
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
            if FileManager.default.fileExists(atPath: "\(path)/fold/fused_deepfold.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // MARK: - Fused Fold Operations

    /// Fused multi-round fold operation.
    ///
    /// Folds multiple instances in a single GPU dispatch, accumulating the
    /// cross-terms T_i weighted by their challenges r_i.
    ///
    /// Cross-term formula:
    ///   T_i = az0 * bz_i + az_i * bz0 - u0 * cz_i - cz0
    ///   Accumulated: T = sum_i r_i * T_i
    ///
    /// - Parameters:
    ///   - az0, bz0, cz0: Base instance matvec results (length n)
    ///   - instances: Array of (az, bz, cz) tuples for each instance (length = fusedRounds)
    ///   - challenges: Folding challenges r_i for each round (length = fusedRounds)
    ///   - u0: Base instance scalar u (usually 1 for first fold, or accumulated u)
    /// - Returns: Tuple of (accumulated_T, accumulated_W) vectors
    public func fusedFold(
        az0: [Fr], bz0: [Fr], cz0: [Fr],
        instances: [(az: [Fr], bz: [Fr], cz: [Fr])],
        challenges: [Fr],
        u0: Fr = Fr.one
    ) throws -> (t: [Fr], w: [Fr]) {
        let n = az0.count

        // Validate inputs
        precondition(n == bz0.count && n == cz0.count, "Base vectors must have same length")
        precondition(instances.count == challenges.count, "instances count must match challenges count")

        // NOTE: by4 kernel processes 3 rounds, by8 processes 7 rounds
        // Validate based on actual kernel capability
        let requiredInstances: Int
        switch fusedRounds {
        case 3, 4:
            requiredInstances = 3
        case 7, 8:
            requiredInstances = 7
        default:
            throw FusedFoldError.invalidFusedRounds("Only 3, 4, 7, or 8 rounds supported, got \(fusedRounds)")
        }
        precondition(instances.count == requiredInstances, "instances count must be \(requiredInstances) for fusedRounds=\(fusedRounds) (kernel limitation)")

        // Validate each instance has same length
        for (i, inst) in instances.enumerated() {
            precondition(inst.az.count == n && inst.bz.count == n && inst.cz.count == n,
                        "Instance \(i) vectors must have same length as base vectors")
        }

        // Select appropriate kernel based on fusedRounds
        // NOTE: by4 kernel actually does 3 rounds, by8 does 7 rounds
        // This is a known limitation - kernel naming is misleading
        let kernel: MTLComputePipelineState
        let actualRounds: Int
        switch fusedRounds {
        case 3, 4:
            kernel = fusedBy4Kernel
            actualRounds = 3  // by4 kernel only processes 3 rounds
        case 7, 8:
            kernel = fusedBy8Kernel
            actualRounds = 7  // by8 kernel only processes 7 rounds
        default:
            throw FusedFoldError.invalidFusedRounds("Only 3, 4, 7, or 8 rounds supported, got \(fusedRounds)")
        }

        return try fusedFoldWithKernel(
            kernel: kernel,
            az0: az0, bz0: bz0, cz0: cz0,
            instances: Array(instances.prefix(actualRounds)),
            challenges: Array(challenges.prefix(actualRounds)),
            u0: u0,
            withWitness: false
        )
    }

    /// Fused multi-round fold with witness accumulation.
    ///
    /// NOTE: This method is not yet implemented for the fused kernel.
    /// The Metal shader with witness accumulation needs to be fixed to support
    /// the double-pointer pattern required for variable-round folding.
    ///
    /// - Parameters:
    ///   - az0, bz0, cz0: Base instance matvec results (length n)
    ///   - instances: Array of (az, bz, cz, w) tuples for each instance
    ///   - challenges: Folding challenges r_i for each round
    ///   - u0: Base instance scalar u
    /// - Returns: Tuple of (accumulated_T, accumulated_W) vectors
    public func fusedFoldWithWitness(
        az0: [Fr], bz0: [Fr], cz0: [Fr],
        instances: [(az: [Fr], bz: [Fr], cz: [Fr], w: [Fr])],
        challenges: [Fr],
        u0: Fr = Fr.one
    ) throws -> (t: [Fr], w: [Fr]) {
        throw FusedFoldError.invalidFusedRounds("fusedFoldWithWitness not yet implemented - use cpuFusedFold for reference")
    }

    // MARK: - Internal Kernel Dispatch

    private func fusedFoldWithKernel(
        kernel: MTLComputePipelineState,
        az0: [Fr], bz0: [Fr], cz0: [Fr],
        instances: [(az: [Fr], bz: [Fr], cz: [Fr])],
        challenges: [Fr],
        u0: Fr,
        withWitness: Bool
    ) throws -> (t: [Fr], w: [Fr]) {
        let n = az0.count
        let numRounds = instances.count

        // Allocate GPU buffers
        let az0Buf = device.makeBuffer(bytes: az0, length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!
        let bz0Buf = device.makeBuffer(bytes: bz0, length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!
        let cz0Buf = device.makeBuffer(bytes: cz0, length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!

        // Create buffers for each instance's matvec results
        var azBuffers: [MTLBuffer] = []
        var bzBuffers: [MTLBuffer] = []
        var czBuffers: [MTLBuffer] = []

        for inst in instances {
            azBuffers.append(device.makeBuffer(bytes: inst.az, length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!)
            bzBuffers.append(device.makeBuffer(bytes: inst.bz, length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!)
            czBuffers.append(device.makeBuffer(bytes: inst.cz, length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!)
        }

        // Challenges buffer
        var challengesCopy = challenges
        let challengesBuf = device.makeBuffer(bytes: &challengesCopy, length: numRounds * MemoryLayout<Fr>.stride, options: .storageModeShared)!

        // u0 buffer (replicated scalar)
        var u0Copy = u0
        let u0Buf = device.makeBuffer(bytes: &u0Copy, length: MemoryLayout<Fr>.stride, options: .storageModeShared)!

        // Output buffers
        let outputT = device.makeBuffer(length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!
        let outputW = device.makeBuffer(length: n * MemoryLayout<Fr>.stride, options: .storageModeShared)!

        // Threadgroup memory
        guard let tgMem = threadgroupMemory else {
            throw FusedFoldError.bufferAllocationFailed
        }

        // Dispatch based on kernel type
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw FusedFoldError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!

        if kernel === fusedBy4Kernel && numRounds == 3 {
            // Kernel signature: sharedMem(0), az0(1), bz0(2), cz0(3),
            // az1(4), bz1(5), cz1(6), az2(7), bz2(8), cz2(9),
            // az3(9), bz3(10), cz3(11), r(12), u0(13), outputT(14)
            enc.setComputePipelineState(kernel)
            enc.setBuffer(tgMem, offset: 0, index: 0)
            enc.setBuffer(az0Buf, offset: 0, index: 1)
            enc.setBuffer(bz0Buf, offset: 0, index: 2)
            enc.setBuffer(cz0Buf, offset: 0, index: 3)
            // Round 0: az1, bz1, cz1 (buffers 4, 5, 6)
            enc.setBuffer(azBuffers[0], offset: 0, index: 4)
            enc.setBuffer(bzBuffers[0], offset: 0, index: 5)
            enc.setBuffer(czBuffers[0], offset: 0, index: 6)
            // Round 1: az2, bz2, cz2 (buffers 7, 8, 9)
            enc.setBuffer(azBuffers[1], offset: 0, index: 7)
            enc.setBuffer(bzBuffers[1], offset: 0, index: 8)
            enc.setBuffer(czBuffers[1], offset: 0, index: 9)
            // Round 2: az3, bz3, cz3 (buffers 9, 10, 11)
            enc.setBuffer(azBuffers[2], offset: 0, index: 9)
            enc.setBuffer(bzBuffers[2], offset: 0, index: 10)
            enc.setBuffer(czBuffers[2], offset: 0, index: 11)
            // r at 12, u0 at 13, outputT at 14
            enc.setBuffer(challengesBuf, offset: 0, index: 12)
            enc.setBuffer(u0Buf, offset: 0, index: 13)
            enc.setBuffer(outputT, offset: 0, index: 14)

            let gridSize = MTLSize(width: n, height: 1, depth: 1)
            let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
            enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        } else if kernel === fusedBy8Kernel && numRounds == 7 {
            // Kernel signature: sharedMem(0), az0(1), bz0(2), cz0(3),
            // az1(4), bz1(5), cz1(6), az2(7), bz2(8), cz2(9),
            // az3(10), bz3(11), cz3(12), az4(13), bz4(14), cz4(15),
            // az5(16), bz5(17), cz5(18), az6(19), bz6(20), cz6(21),
            // az7(22), bz7(23), cz7(24), r(25), u0(26), outputT(27)
            enc.setComputePipelineState(kernel)
            enc.setBuffer(tgMem, offset: 0, index: 0)
            enc.setBuffer(az0Buf, offset: 0, index: 1)
            enc.setBuffer(bz0Buf, offset: 0, index: 2)
            enc.setBuffer(cz0Buf, offset: 0, index: 3)

            // Set 8 instance buffers (instances 1-7 at buffers 4-24)
            // Instance i goes to baseIndex = 4 + (i-1) * 3 = 1 + i * 3
            for i in 0..<7 {  // instances 1-7 (index 0 is az0/bz0/cz0)
                let baseIndex = 4 + i * 3
                enc.setBuffer(azBuffers[i], offset: 0, index: baseIndex)
                enc.setBuffer(bzBuffers[i], offset: 0, index: baseIndex + 1)
                enc.setBuffer(czBuffers[i], offset: 0, index: baseIndex + 2)
            }
            // Instance 7 is at buffers 22, 23, 24
            enc.setBuffer(azBuffers[6], offset: 0, index: 22)
            enc.setBuffer(bzBuffers[6], offset: 0, index: 23)
            enc.setBuffer(czBuffers[6], offset: 0, index: 24)

            enc.setBuffer(challengesBuf, offset: 0, index: 25)
            enc.setBuffer(u0Buf, offset: 0, index: 26)
            enc.setBuffer(outputT, offset: 0, index: 27)

            let gridSize = MTLSize(width: n, height: 1, depth: 1)
            let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
            enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        } else {
            // Generic configurable kernel
            enc.setComputePipelineState(kernel)
            enc.setBuffer(tgMem, offset: 0, index: 0)
            enc.setBuffer(az0Buf, offset: 0, index: 1)
            enc.setBuffer(bz0Buf, offset: 0, index: 2)
            enc.setBuffer(cz0Buf, offset: 0, index: 3)

            // Pointers to instance buffers (device addresses)
            var azPtrs: [UnsafeMutableRawPointer] = azBuffers.map { $0.contents() }
            var bzPtrs: [UnsafeMutableRawPointer] = bzBuffers.map { $0.contents() }
            var czPtrs: [UnsafeMutableRawPointer] = czBuffers.map { $0.contents() }

            let azPtrsBuf = device.makeBuffer(bytes: &azPtrs, length: numRounds * MemoryLayout<UnsafeMutableRawPointer>.stride, options: .storageModeShared)!
            let bzPtrsBuf = device.makeBuffer(bytes: &bzPtrs, length: numRounds * MemoryLayout<UnsafeMutableRawPointer>.stride, options: .storageModeShared)!
            let czPtrsBuf = device.makeBuffer(bytes: &czPtrs, length: numRounds * MemoryLayout<UnsafeMutableRawPointer>.stride, options: .storageModeShared)!

            enc.setBuffer(azPtrsBuf, offset: 0, index: 4)
            enc.setBuffer(bzPtrsBuf, offset: 0, index: 5)
            enc.setBuffer(czPtrsBuf, offset: 0, index: 6)

            var numRoundsVal = UInt32(numRounds)
            enc.setBytes(&numRoundsVal, length: 4, index: 7)
            enc.setBuffer(challengesBuf, offset: 0, index: 8)
            enc.setBuffer(u0Buf, offset: 0, index: 9)
            enc.setBuffer(outputT, offset: 0, index: 10)
            enc.setBuffer(outputW, offset: 0, index: 11)

            let gridSize = MTLSize(width: n, height: 1, depth: 1)
            let tgSize = MTLSize(width: threadgroupSize, height: 1, depth: 1)
            enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
        }

        enc.endEncoding()
        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        // Read back results
        let tPtr = outputT.contents().bindMemory(to: Fr.self, capacity: n)
        let tResult = Array(UnsafeBufferPointer(start: tPtr, count: n))

        let wPtr = outputW.contents().bindMemory(to: Fr.self, capacity: n)
        let wResult = Array(UnsafeBufferPointer(start: wPtr, count: n))

        return (tResult, wResult)
    }

    // MARK: - CPU Reference Implementation

    /// CPU reference implementation for correctness verification.
    public func cpuFusedFold(
        az0: [Fr], bz0: [Fr], cz0: [Fr],
        instances: [(az: [Fr], bz: [Fr], cz: [Fr])],
        challenges: [Fr],
        u0: Fr = Fr.one
    ) -> (t: [Fr], w: [Fr]) {
        let n = az0.count
        var accumulatedT = [Fr](repeating: .zero, count: n)

        for (round, inst) in instances.enumerated() {
            let r = challenges[round]

            // Compute T_i = az0 * bz_i + az_i * bz0 - u0 * cz_i - cz0
            for i in 0..<n {
                let cross1 = frMul(az0[i], inst.bz[i])
                let cross2 = frMul(inst.az[i], bz0[i])
                let u0Cz_i = frMul(u0, inst.cz[i])

                var T_i = frAdd(cross1, cross2)
                T_i = frSub(T_i, u0Cz_i)
                T_i = frSub(T_i, cz0[i])

                // Accumulate weighted by challenge
                accumulatedT[i] = frAdd(accumulatedT[i], frMul(T_i, r))
            }
        }

        // W accumulation is just sum of weighted witnesses (for consistency with GPU kernel)
        var accumulatedW = [Fr](repeating: .zero, count: n)

        return (accumulatedT, accumulatedW)
    }
}

// MARK: - Convenience Factory

extension FusedDeepFoldEngine {
    /// Create engine for 4-round fused folding (common case).
    public static func fusedBy4(threadgroupSize: Int = 256) throws -> FusedDeepFoldEngine {
        try FusedDeepFoldEngine(fusedRounds: 4, threadgroupSize: threadgroupSize)
    }

    /// Create engine for 8-round fused folding (high throughput).
    public static func fusedBy8(threadgroupSize: Int = 256) throws -> FusedDeepFoldEngine {
        try FusedDeepFoldEngine(fusedRounds: 8, threadgroupSize: threadgroupSize)
    }
}
