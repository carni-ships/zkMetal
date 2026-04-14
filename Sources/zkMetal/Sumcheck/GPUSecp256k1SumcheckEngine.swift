// GPU-Accelerated Fused Sumcheck Engine for secp256k1 Folding
//
// Implements GPU-dispatched fused sumcheck rounds for secp256k1 folding schemes.
// The fused kernel combines eq-computation, weighting, and fold accumulation
// in a single GPU dispatch, eliminating intermediate memory round-trips.
//
// Folding schemes supported:
//   - Nova: sumcheck over relaxed R1CS instances
//   - HyperNova: CCS-based folding with multi-table sumcheck
//   - Supernova: multi-circuit folding
//
// Reference: "Nova: Recursive Zero-Knowledge Arguments from Folding Schemes"
//            (Kothapalli, Setty, Tzialla 2022)

import Foundation
import Metal
import NeonFieldOps

// MARK: - Error Type

public enum GPUSecp256k1SumcheckError: Error {
    case noGPU
    case noCommandQueue
    case noCommandBuffer
    case missingKernel(String)
    case gpuDispatchFailed(String)
    case invalidInput(String)
}

// MARK: - Kernel Type

/// Available GPU kernels for secp256k1 sumcheck
public enum Secp256k1SumcheckKernel {
    /// Fused eq-compute + fold accumulate (single-table)
    case fusedSumcheckFold
    /// Fused eq-compute + fold with sequential variables
    case fusedSumcheckFoldReg
    /// Fused multi-table sumcheck fold (CCS/HyperNova style)
    case fusedMultiTableFold
    /// Standard fused round poly + reduce
    case fusedRoundReduce
    /// Final reduction of partial sums
    case finalReduce
}

// MARK: - GPU Secp256k1 Sumcheck Engine

/// GPU-accelerated fused sumcheck engine for secp256k1 folding.
///
/// This engine dispatches the fused sumcheck kernels that combine:
///   1. eq(tau, x) inner product computation
///   2. Weighting by fold challenges
///   3. Accumulation into fold accumulator
///
/// All in a single GPU dispatch, eliminating memory round-trips between
/// separate eq-compute and fold steps.
///
/// Usage:
///   1. Create engine with `GPUSecp256k1SumcheckEngine()`
///   2. Call fusedSumcheckFold() for single-table folding
///   3. Or call fusedMultiTableFold() for CCS/HyperNova-style folding
public final class GPUSecp256k1SumcheckEngine {

    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue

    // Kernel pipeline states
    private let fusedSumcheckFoldPipeline: MTLComputePipelineState
    private let fusedSumcheckFoldRegPipeline: MTLComputePipelineState
    private let fusedMultiTableFoldPipeline: MTLComputePipelineState
    private let fusedRoundReducePipeline: MTLComputePipelineState
    private let finalReducePipeline: MTLComputePipelineState

    // Cached buffers for reuse
    private var evalBufferA: MTLBuffer?
    private var evalBufferB: MTLBuffer?
    private var evalBufferCapacity: Int = 0
    private var partialBuffer: MTLBuffer?
    private var partialBufferCapacity: Int = 0

    // CPU fallback threshold
    private static let gpuThreshold = 4096

    // MARK: - Initialization

    /// Initialize the GPU secp256k1 sumcheck engine.
    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw GPUSecp256k1SumcheckError.noGPU
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw GPUSecp256k1SumcheckError.noCommandQueue
        }
        self.commandQueue = queue

        let library = try GPUSecp256k1SumcheckEngine.compileShaders(device: device)

        // Load all kernel pipelines
        guard let fusedFold = library.makeFunction(name: "secp_fused_sumcheck_fold"),
              let fusedFoldReg = library.makeFunction(name: "secp_fused_sumcheck_fold_reg"),
              let fusedMultiFold = library.makeFunction(name: "secp_fused_sumcheck_multitable_fold"),
              let fusedRound = library.makeFunction(name: "secp_fused_round_reduce"),
              let finalReduce = library.makeFunction(name: "secp_sumcheck_final_reduce") else {
            let missing = [
                "secp_fused_sumcheck_fold",
                "secp_fused_sumcheck_fold_reg",
                "secp_fused_sumcheck_multitable_fold",
                "secp_fused_round_reduce",
                "secp_sumcheck_final_reduce"
            ].filter { library.makeFunction(name: $0) == nil }
            throw GPUSecp256k1SumcheckError.missingKernel("Missing kernels: \(missing)")
        }

        self.fusedSumcheckFoldPipeline = try device.makeComputePipelineState(function: fusedFold)
        self.fusedSumcheckFoldRegPipeline = try device.makeComputePipelineState(function: fusedFoldReg)
        self.fusedMultiTableFoldPipeline = try device.makeComputePipelineState(function: fusedMultiFold)
        self.fusedRoundReducePipeline = try device.makeComputePipelineState(function: fusedRound)
        self.finalReducePipeline = try device.makeComputePipelineState(function: finalReduce)
    }

    // MARK: - Shader Compilation

    private static func compileShaders(device: MTLDevice) throws -> MTLLibrary {
        let shaderDir = findShaderDir()

        // Load secp256k1 field arithmetic
        let fpSource = try String(contentsOfFile: shaderDir + "/fields/secp256k1_fp.metal", encoding: .utf8)
        let cleanFp = fpSource
            .replacingOccurrences(of: "#ifndef SECP256K1_FP_METAL", with: "")
            .replacingOccurrences(of: "#define SECP256K1_FP_METAL", with: "")
            .replacingOccurrences(of: "#endif // SECP256K1_FP_METAL", with: "")

        // Load sumcheck kernels
        let sumcheckSource = try String(contentsOfFile: shaderDir + "/sumcheck/secp256k1_sumcheck.metal", encoding: .utf8)

        let combined = cleanFp + "\n" + sumcheckSource
        let options = MTLCompileOptions()
        options.fastMathEnabled = true

        return try device.makeLibrary(source: combined, options: options)
    }

    private static func findShaderDir() -> String {
        let execPath = CommandLine.arguments[0]
        let execDir = (execPath as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                let path = url.appendingPathComponent("fields/secp256k1_fp.metal").path
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
            if FileManager.default.fileExists(atPath: "\(path)/fields/secp256k1_fp.metal") {
                return path
            }
        }
        return "./Sources/Shaders"
    }

    // MARK: - Buffer Management

    private func ensureEvalBuffers(byteCount: Int) {
        if evalBufferCapacity >= byteCount { return }
        evalBufferA = device.makeBuffer(length: byteCount, options: .storageModeShared)
        evalBufferB = device.makeBuffer(length: byteCount, options: .storageModeShared)
        evalBufferCapacity = byteCount
    }

    private func ensurePartialBuffer(byteCount: Int) {
        if partialBufferCapacity >= byteCount { return }
        partialBuffer = device.makeBuffer(length: byteCount, options: .storageModeShared)
        partialBufferCapacity = byteCount
    }

    // MARK: - Fused Sumcheck Fold (Single Table)

    /// Compute fused sumcheck fold: accumulator = sum_i eq(tau, i) * eval[i] * r
    ///
    /// This combines eq-computation, weighting, and accumulation in one GPU dispatch.
    /// Single-table version for standard Nova-style folding.
    ///
    /// - Parameters:
    ///   - evals: Evaluation table g(x) for x in {0,1}^s
    ///   - tau: Point to evaluate eq polynomial at
    ///   - challenge: Fold challenge r
    /// - Returns: Accumulated result
    public func fusedSumcheckFold(
        evals: [SecpFp],
        tau: [SecpFp],
        challenge: SecpFp
    ) throws -> SecpFp {
        let n = evals.count
        let s = tau.count

        precondition(n == (1 << s), "Domain size must be power of 2")

        // CPU fallback for small inputs
        if n < Self.gpuThreshold {
            return cpuFusedSumcheckFold(evals: evals, tau: tau, challenge: challenge)
        }

        let stride = MemoryLayout<SecpFp>.stride

        // Allocate buffers
        guard let evalBuf = device.makeBuffer(length: n * stride, options: .storageModeShared),
              let accBuf = device.makeBuffer(length: stride, options: .storageModeShared) else {
            throw GPUSecp256k1SumcheckError.gpuDispatchFailed("Buffer allocation failed")
        }

        // Copy input data
        evals.withUnsafeBytes { src in
            memcpy(evalBuf.contents(), src.baseAddress!, n * stride)
        }
        // Challenge is set via setBytes below

        let tgSize = 256
        let numGroups = max(1, (n + tgSize - 1) / tgSize)

        // Dispatch kernel
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw GPUSecp256k1SumcheckError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fusedSumcheckFoldPipeline)
        enc.setBuffer(evalBuf, offset: 0, index: 0)
        enc.setBuffer(accBuf, offset: 0, index: 1)

        // Challenge (fold weight)
        var chal = challenge
        enc.setBytes(&chal, length: stride, index: 2)

        // Tau point
        tau.withUnsafeBytes { tauBuf in
            enc.setBytes(tauBuf.baseAddress!, length: s * stride, index: 3)
        }

        // N and S
        var nVal = UInt32(n)
        var sVal = UInt32(s)
        enc.setBytes(&nVal, length: 4, index: 4)
        enc.setBytes(&sVal, length: 4, index: 5)

        enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw GPUSecp256k1SumcheckError.gpuDispatchFailed(error.localizedDescription)
        }

        // Read result
        let resultPtr = accBuf.contents().bindMemory(to: SecpFp.self, capacity: 1)
        return resultPtr[0]
    }

    // MARK: - Fused Multi-Table Sumcheck Fold (CCS/HyperNova)

    /// Compute fused multi-table sumcheck fold for CCS-style folding
    ///
    /// accumulator = r * sum_t(sum_i eq(tau, i) * table_t[i])
    ///
    /// Used in HyperNova where multiple polynomials are folded together.
    ///
    /// - Parameters:
    ///   - tables: Concatenated evaluation tables (t * n elements)
    ///   - n: Domain size per table
    ///   - t: Number of tables
    ///   - tau: eq evaluation point
    ///   - challenge: Fold challenge r
    /// - Returns: Accumulated result
    public func fusedMultiTableFold(
        tables: [SecpFp],
        n: Int,
        t: Int,
        tau: [SecpFp],
        challenge: SecpFp
    ) throws -> SecpFp {
        precondition(tables.count == n * t, "Tables size mismatch")

        let s = tau.count
        precondition(n == (1 << s), "Domain size must be power of 2")

        // CPU fallback for small inputs
        if n < Self.gpuThreshold {
            return cpuFusedMultiTableFold(tables: tables, n: n, t: t, tau: tau, challenge: challenge)
        }

        let stride = MemoryLayout<SecpFp>.stride

        // Allocate buffers
        guard let tablesBuf = device.makeBuffer(length: tables.count * stride, options: .storageModeShared),
              let accBuf = device.makeBuffer(length: stride, options: .storageModeShared) else {
            throw GPUSecp256k1SumcheckError.gpuDispatchFailed("Buffer allocation failed")
        }

        // Copy input data
        tables.withUnsafeBytes { src in
            memcpy(tablesBuf.contents(), src.baseAddress!, tables.count * stride)
        }

        let tgSize = 256
        let numGroups = max(1, (n + tgSize - 1) / tgSize)

        // Dispatch kernel
        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw GPUSecp256k1SumcheckError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fusedMultiTableFoldPipeline)
        enc.setBuffer(tablesBuf, offset: 0, index: 0)
        enc.setBuffer(accBuf, offset: 0, index: 1)

        var chal = challenge
        enc.setBytes(&chal, length: stride, index: 2)

        tau.withUnsafeBytes { tauBuf in
            enc.setBytes(tauBuf.baseAddress!, length: s * stride, index: 3)
        }

        var nVal = UInt32(n)
        var sVal = UInt32(s)
        var tVal = UInt32(t)
        enc.setBytes(&nVal, length: 4, index: 4)
        enc.setBytes(&sVal, length: 4, index: 5)
        enc.setBytes(&tVal, length: 4, index: 6)

        enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw GPUSecp256k1SumcheckError.gpuDispatchFailed(error.localizedDescription)
        }

        let resultPtr = accBuf.contents().bindMemory(to: SecpFp.self, capacity: 1)
        return resultPtr[0]
    }

    // MARK: - Fused Round Reduce (Standard Sumcheck)

    /// Compute fused round polynomial + reduce
    ///
    /// This is the standard sumcheck kernel that:
    ///   1. Computes partial sums (s0, s1, s2)
    ///   2. Reduces the table by the challenge
    /// Both in one GPU dispatch.
    ///
    /// - Parameters:
    ///   - evals: Input evaluations (size 2*n)
    ///   - challenge: Fold challenge r
    /// - Returns: (reduced_evals, partial_sums)
    public func fusedRoundReduce(
        evals: [SecpFp],
        challenge: SecpFp
    ) throws -> (reduced: [SecpFp], partialSums: [SecpFp]) {
        let n = evals.count / 2
        precondition(evals.count == 2 * n, "Evals size must be 2*n")

        if n < Self.gpuThreshold {
            return cpuFusedRoundReduce(evals: evals, challenge: challenge)
        }

        let stride = MemoryLayout<SecpFp>.stride
        let tgSize = 256
        let numGroups = max(1, (n + tgSize - 1) / tgSize)

        // Allocate buffers
        ensureEvalBuffers(byteCount: n * stride)
        ensurePartialBuffer(byteCount: numGroups * 3 * stride)

        guard let evalOutBuf = evalBufferA,
              let partialBuf = partialBuffer,
              let evalInBuf = device.makeBuffer(length: evals.count * stride, options: .storageModeShared) else {
            throw GPUSecp256k1SumcheckError.gpuDispatchFailed("Buffer allocation failed")
        }

        evals.withUnsafeBytes { src in
            memcpy(evalInBuf.contents(), src.baseAddress!, evals.count * stride)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw GPUSecp256k1SumcheckError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(fusedRoundReducePipeline)
        enc.setBuffer(evalInBuf, offset: 0, index: 0)
        enc.setBuffer(evalOutBuf, offset: 0, index: 1)
        enc.setBuffer(partialBuf, offset: 0, index: 2)

        var chal = challenge
        enc.setBytes(&chal, length: stride, index: 3)
        var nVal = UInt32(n)
        enc.setBytes(&nVal, length: 4, index: 4)

        enc.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw GPUSecp256k1SumcheckError.gpuDispatchFailed(error.localizedDescription)
        }

        // Read reduced evals
        let reduced = [SecpFp](repeating: SecpFp.zero, count: n)
        var reducedArray = reduced
        reducedArray.withUnsafeMutableBytes { dst in
            memcpy(dst.baseAddress!, evalOutBuf.contents(), n * stride)
        }

        // Read partial sums
        let partialSums = [SecpFp](repeating: SecpFp.zero, count: numGroups * 3)
        var partialArray = partialSums
        partialArray.withUnsafeMutableBytes { dst in
            memcpy(dst.baseAddress!, partialBuf.contents(), numGroups * 3 * stride)
        }

        return (reduced: reducedArray, partialSums: partialArray)
    }

    /// Final reduction of partial sums
    public func finalReduce(partialSums: [SecpFp]) throws -> (SecpFp, SecpFp, SecpFp) {
        let numGroups = partialSums.count / 3
        precondition(partialSums.count == numGroups * 3, "Partial sums size must be 3 * numGroups")

        let stride = MemoryLayout<SecpFp>.stride
        let tgSize = 256

        ensurePartialBuffer(byteCount: partialSums.count * stride)
        guard let partialInBuf = partialBuffer,
              let outputBuf = device.makeBuffer(length: 3 * stride, options: .storageModeShared) else {
            throw GPUSecp256k1SumcheckError.gpuDispatchFailed("Buffer allocation failed")
        }

        partialSums.withUnsafeBytes { src in
            memcpy(partialInBuf.contents(), src.baseAddress!, partialSums.count * stride)
        }

        guard let cmdBuf = commandQueue.makeCommandBuffer() else {
            throw GPUSecp256k1SumcheckError.noCommandBuffer
        }

        let enc = cmdBuf.makeComputeCommandEncoder()!
        enc.setComputePipelineState(finalReducePipeline)
        enc.setBuffer(partialInBuf, offset: 0, index: 0)
        enc.setBuffer(outputBuf, offset: 0, index: 1)
        var ngVal = UInt32(numGroups)
        enc.setBytes(&ngVal, length: 4, index: 2)

        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        enc.endEncoding()

        cmdBuf.commit()
        cmdBuf.waitUntilCompleted()

        if let error = cmdBuf.error {
            throw GPUSecp256k1SumcheckError.gpuDispatchFailed(error.localizedDescription)
        }

        let outputPtr = outputBuf.contents().bindMemory(to: SecpFp.self, capacity: 3)
        return (outputPtr[0], outputPtr[1], outputPtr[2])
    }

    // MARK: - CPU Reference Implementations

    /// CPU reference for fused sumcheck fold (single-table)
    private func cpuFusedSumcheckFold(
        evals: [SecpFp],
        tau: [SecpFp],
        challenge: SecpFp
    ) -> SecpFp {
        let n = evals.count
        let s = tau.count

        var accumulator = SecpFp.zero

        for idx in 0..<n {
            // Compute eq(tau, idx)
            var eqVal = SecpFp.one
            var temp = idx
            for i in 0..<s {
                let bit = temp & 1
                let ti = tau[i]
                let oneMinusTi = secpSub(SecpFp.one, ti)
                let eqI: SecpFp
                if bit == 0 {
                    eqI = oneMinusTi
                } else {
                    eqI = ti
                }
                eqVal = secpMul(eqVal, eqI)
                temp >>= 1
            }

            // contribution = eqVal * evals[idx]
            let contribution = secpMul(eqVal, evals[idx])
            // weighted = challenge * contribution
            let weighted = secpMul(challenge, contribution)
            // accumulator += weighted
            accumulator = secpAdd(accumulator, weighted)
        }

        return accumulator
    }

    /// CPU reference for fused multi-table fold
    private func cpuFusedMultiTableFold(
        tables: [SecpFp],
        n: Int,
        t: Int,
        tau: [SecpFp],
        challenge: SecpFp
    ) -> SecpFp {
        let s = tau.count
        var accumulator = SecpFp.zero

        for idx in 0..<n {
            // Compute eq(tau, idx)
            var eqVal = SecpFp.one
            var temp = idx
            for i in 0..<s {
                let bit = temp & 1
                let ti = tau[i]
                let oneMinusTi = secpSub(SecpFp.one, ti)
                let eqI: SecpFp = (bit == 0) ? oneMinusTi : ti
                eqVal = secpMul(eqVal, eqI)
                temp >>= 1
            }

            // Sum over all t tables
            var tableSum = SecpFp.zero
            for tableIdx in 0..<t {
                let flatIdx = tableIdx * n + idx
                let contribution = secpMul(eqVal, tables[flatIdx])
                tableSum = secpAdd(tableSum, contribution)
            }

            let weighted = secpMul(challenge, tableSum)
            accumulator = secpAdd(accumulator, weighted)
        }

        return accumulator
    }

    /// CPU reference for fused round reduce
    private func cpuFusedRoundReduce(
        evals: [SecpFp],
        challenge: SecpFp
    ) -> (reduced: [SecpFp], partialSums: [SecpFp]) {
        let n = evals.count / 2
        var reduced = [SecpFp](repeating: SecpFp.zero, count: n)
        var s0 = SecpFp.zero
        var s1 = SecpFp.zero
        var s2 = SecpFp.zero

        for i in 0..<n {
            let a = evals[i]
            let b = evals[i + n]

            // Partial sums
            s0 = secpAdd(s0, a)
            s1 = secpAdd(s1, b)
            // f(2) = 2*f(1) - f(0)
            let twoB = secpDouble(b)
            let f2 = secpSub(twoB, a)
            s2 = secpAdd(s2, f2)

            // Reduced value
            let diff = secpSub(b, a)
            let rDiff = secpMul(challenge, diff)
            reduced[i] = secpAdd(a, rDiff)
        }

        return (reduced: reduced, partialSums: [s0, s1, s2])
    }
}
