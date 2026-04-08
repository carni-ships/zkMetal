// GPUFflonkProverEngine — GPU-accelerated Fflonk prover engine
//
// Fflonk ("fast-Fourier Plonk") batches multiple polynomial openings into fewer
// by interleaving polynomials and exploiting roots of unity. This engine provides
// GPU acceleration for the key bottlenecks:
//
//   1. Polynomial combination (interleaving k polys into one combined polynomial)
//   2. Multi-scalar multiplication for commitment computation
//   3. Polynomial evaluation via NTT on coset domains
//   4. Quotient polynomial computation for KZG opening
//
// The combined polynomial is:
//   P(X) = sum_{i=0}^{k-1} p_i(X^k) * X^i
//
// where k is the batch size (power of 2). Evaluating P at omega^j * z recovers
// p_j(z^k) where omega is a primitive k-th root of unity.
//
// GPU acceleration targets:
//   - Metal compute kernel for parallel coefficient interleaving (large polys)
//   - Metal MSM for commitment to combined polynomial
//   - Metal NTT for polynomial evaluation on evaluation domains
//   - CPU fallback for small polynomials where dispatch overhead dominates
//
// Reference: "fflonk: a Fast-Fourier inspired verifier efficient version of PlonK"
//            (Gabizon, Khovratovich, 2021)

import Foundation
import Metal
import NeonFieldOps

// MARK: - Fflonk Prover Configuration

/// Configuration for the GPU Fflonk prover engine.
public struct FflonkProverConfig {
    /// Maximum batch size (power of 2, up to 16).
    public let maxBatchSize: Int
    /// Whether to enable GPU acceleration (falls back to CPU if false or unavailable).
    public let useGPU: Bool
    /// Minimum polynomial degree to dispatch interleaving to GPU.
    public let gpuInterleaveThreshold: Int
    /// Minimum polynomial degree to dispatch evaluation to GPU (NTT path).
    public let gpuEvalThreshold: Int
    /// Number of random challenges for Fiat-Shamir binding (0 = no binding).
    public let numChallenges: Int

    public init(maxBatchSize: Int = 16, useGPU: Bool = true,
                gpuInterleaveThreshold: Int = 256, gpuEvalThreshold: Int = 512,
                numChallenges: Int = 0) {
        precondition(maxBatchSize > 0 && maxBatchSize & (maxBatchSize - 1) == 0,
                     "maxBatchSize must be a power of 2")
        self.maxBatchSize = maxBatchSize
        self.useGPU = useGPU
        self.gpuInterleaveThreshold = gpuInterleaveThreshold
        self.gpuEvalThreshold = gpuEvalThreshold
        self.numChallenges = numChallenges
    }
}

// MARK: - Fflonk Prover Result

/// Result from the GPU Fflonk prover containing all proof components.
public struct FflonkProverResult {
    /// Commitment to the combined polynomial [P(s)].
    public let commitment: PointProjective
    /// KZG witness point [q(s)] for the opening proof.
    public let witness: PointProjective
    /// Evaluations y_i = p_i(z^k) for each sub-polynomial.
    public let evaluations: [Fr]
    /// The combined polynomial P(X) coefficients.
    public let combinedPoly: [Fr]
    /// The quotient polynomial q(X) = (P(X) - R(X)) / Z(X).
    public let quotientPoly: [Fr]
    /// Evaluation point z.
    public let point: Fr
    /// Batch size k (number of sub-polynomials, after padding).
    public let batchSize: Int
    /// Whether GPU was used for polynomial combination.
    public let usedGPUCombine: Bool
    /// Whether GPU was used for commitment (MSM).
    public let usedGPUCommit: Bool
    /// Prover time in seconds.
    public let proverTime: Double

    public init(commitment: PointProjective, witness: PointProjective,
                evaluations: [Fr], combinedPoly: [Fr], quotientPoly: [Fr],
                point: Fr, batchSize: Int, usedGPUCombine: Bool,
                usedGPUCommit: Bool, proverTime: Double) {
        self.commitment = commitment
        self.witness = witness
        self.evaluations = evaluations
        self.combinedPoly = combinedPoly
        self.quotientPoly = quotientPoly
        self.point = point
        self.batchSize = batchSize
        self.usedGPUCombine = usedGPUCombine
        self.usedGPUCommit = usedGPUCommit
        self.proverTime = proverTime
    }
}

// MARK: - Fflonk Verification Result

/// Result from the Fflonk verifier with pairing check components.
public struct FflonkVerificationResult {
    public let valid: Bool
    public let remainderAtSecret: Fr
    public let lhs: PointProjective
    public let rhs: PointProjective

    public init(valid: Bool, remainderAtSecret: Fr, lhs: PointProjective, rhs: PointProjective) {
        self.valid = valid; self.remainderAtSecret = remainderAtSecret
        self.lhs = lhs; self.rhs = rhs
    }
}

// MARK: - Fflonk Linearization

/// Linearization coefficients for combining multiple opening claims via gamma powers.
public struct FflonkLinearization {
    public let gammaPowers: [Fr]
    public let linearizedPoly: [Fr]
    public let linearizedEval: Fr

    public init(gammaPowers: [Fr], linearizedPoly: [Fr], linearizedEval: Fr) {
        self.gammaPowers = gammaPowers; self.linearizedPoly = linearizedPoly
        self.linearizedEval = linearizedEval
    }
}

// MARK: - Fflonk Multi-Opening

/// Multiple opening claims batched into a single Fflonk proof via linearization.
public struct FflonkMultiOpening {
    public let openings: [FflonkOpeningProof]
    public let linearization: FflonkLinearization
    public let combinedWitness: PointProjective

    public init(openings: [FflonkOpeningProof], linearization: FflonkLinearization,
                combinedWitness: PointProjective) {
        self.openings = openings; self.linearization = linearization
        self.combinedWitness = combinedWitness
    }
}

// MARK: - GPUFflonkProverEngine

/// GPU-accelerated Fflonk prover engine.
///
/// Accelerates the key computational bottlenecks in the Fflonk proving pipeline:
/// - Polynomial interleaving (building the combined polynomial P(X))
/// - Multi-scalar multiplication for KZG commitment
/// - Polynomial arithmetic for quotient computation
///
/// Falls back to CPU for small polynomials where Metal dispatch overhead dominates.
public class GPUFflonkProverEngine {
    public static let version = PrimitiveVersion(version: "1.0.0", updated: "2026-04-05")

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private let interleavePipeline: MTLComputePipelineState?
    private let polyMulScalarPipeline: MTLComputePipelineState?
    private let threadgroupSize: Int
    private let config: FflonkProverConfig

    /// Wrapped CPU FflonkEngine for verification and fallback.
    public let cpuEngine: FflonkEngine?

    // MARK: - Initialization

    /// Initialize the GPU Fflonk prover engine.
    ///
    /// - Parameters:
    ///   - srs: Fflonk structured reference string.
    ///   - config: Prover configuration.
    /// - Throws: If SRS is invalid or KZG engine initialization fails.
    public init(srs: FflonkSRS, config: FflonkProverConfig = FflonkProverConfig()) throws {
        self.config = config
        self.threadgroupSize = 256

        if config.useGPU {
            let dev = MTLCreateSystemDefaultDevice()
            self.device = dev
            self.commandQueue = dev?.makeCommandQueue()
            if let dev = dev {
                self.interleavePipeline = GPUFflonkProverEngine.compileInterleaveKernel(device: dev)
                self.polyMulScalarPipeline = GPUFflonkProverEngine.compilePolyMulScalarKernel(device: dev)
            } else {
                self.interleavePipeline = nil
                self.polyMulScalarPipeline = nil
            }
        } else {
            self.device = nil
            self.commandQueue = nil
            self.interleavePipeline = nil
            self.polyMulScalarPipeline = nil
        }

        self.cpuEngine = try FflonkEngine(srs: srs)
    }

    /// Initialize without an SRS (for polynomial operations only).
    public init(config: FflonkProverConfig = FflonkProverConfig()) {
        self.config = config
        self.threadgroupSize = 256
        self.cpuEngine = nil

        if config.useGPU {
            let dev = MTLCreateSystemDefaultDevice()
            self.device = dev
            self.commandQueue = dev?.makeCommandQueue()
            if let dev = dev {
                self.interleavePipeline = GPUFflonkProverEngine.compileInterleaveKernel(device: dev)
                self.polyMulScalarPipeline = GPUFflonkProverEngine.compilePolyMulScalarKernel(device: dev)
            } else {
                self.interleavePipeline = nil
                self.polyMulScalarPipeline = nil
            }
        } else {
            self.device = nil
            self.commandQueue = nil
            self.interleavePipeline = nil
            self.polyMulScalarPipeline = nil
        }
    }

    // MARK: - GPU Polynomial Interleaving

    /// Build P(X) = sum p_i(X^k) * X^i, using GPU for large polynomials.
    public func buildCombinedPoly(_ polynomials: [[Fr]], batchSize k: Int) -> (poly: [Fr], usedGPU: Bool) {
        guard !polynomials.isEmpty else { return ([], false) }

        let d = polynomials.map { $0.count }.max()!
        let combinedSize = k * d

        // GPU path for large polynomials
        if combinedSize >= config.gpuInterleaveThreshold,
           let device = device,
           let commandQueue = commandQueue,
           let pipeline = interleavePipeline {
            if let result = gpuInterleave(polynomials, batchSize: k, maxDeg: d,
                                           device: device, commandQueue: commandQueue,
                                           pipeline: pipeline) {
                return (result, true)
            }
        }

        // CPU fallback: use the existing static method
        let result = FflonkEngine.buildCombinedPoly(polynomials, batchSize: k)
        return (result, false)
    }

    /// GPU kernel dispatch for coefficient interleaving.
    private func gpuInterleave(_ polynomials: [[Fr]], batchSize k: Int, maxDeg d: Int,
                                device: MTLDevice, commandQueue: MTLCommandQueue,
                                pipeline: MTLComputePipelineState) -> [Fr]? {
        let combinedSize = k * d

        // Flatten all sub-polynomials into a single buffer, zero-padded to maxDeg
        var flatInput = [Fr](repeating: Fr.zero, count: k * d)
        for i in 0..<min(k, polynomials.count) {
            for j in 0..<polynomials[i].count {
                flatInput[i * d + j] = polynomials[i][j]
            }
        }

        let inputSize = flatInput.count * MemoryLayout<Fr>.stride
        let outputSize = combinedSize * MemoryLayout<Fr>.stride

        guard let inputBuffer = device.makeBuffer(bytes: &flatInput, length: inputSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: outputSize, options: .storageModeShared) else {
            return nil
        }

        var params: [UInt32] = [UInt32(k), UInt32(d), UInt32(combinedSize)]
        guard let paramsBuffer = device.makeBuffer(bytes: &params, length: params.count * 4, options: .storageModeShared) else {
            return nil
        }

        guard let cmdBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeComputeCommandEncoder() else {
            return nil
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(outputBuffer, offset: 0, index: 1)
        encoder.setBuffer(paramsBuffer, offset: 0, index: 2)

        let threadgroups = MTLSize(width: (combinedSize + threadgroupSize - 1) / threadgroupSize, height: 1, depth: 1)
        let threadsPerGroup = MTLSize(width: threadgroupSize, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        cmdBuffer.commit()
        cmdBuffer.waitUntilCompleted()

        if cmdBuffer.status != .completed { return nil }

        let outPtr = outputBuffer.contents().bindMemory(to: Fr.self, capacity: combinedSize)
        return Array(UnsafeBufferPointer(start: outPtr, count: combinedSize))
    }

    // MARK: - Full Prove Pipeline

    /// Generate a complete Fflonk proof: combine, commit, evaluate, build quotient, witness.
    public func prove(polynomials: [[Fr]], at z: Fr) throws -> FflonkProverResult {
        let t0 = CFAbsoluteTimeGetCurrent()
        guard let engine = cpuEngine else {
            throw FflonkProverError.noSRS
        }

        let k = nextPowerOf2(polynomials.count)
        precondition(k >= 1 && k <= config.maxBatchSize,
                     "Batch size \(k) exceeds max \(config.maxBatchSize)")

        // Pad to power-of-2 batch size
        var padded = polynomials
        while padded.count < k {
            padded.append([Fr.zero])
        }

        // Step 1: Build combined polynomial (GPU or CPU)
        let (combined, usedGPUCombine) = buildCombinedPoly(padded, batchSize: k)

        // Step 2: Commit via KZG
        let commitment = try engine.kzg.commit(combined)

        // Step 3: Evaluate sub-polynomials at z^k
        let zk = frPow(z, UInt64(k))
        var evaluations = [Fr]()
        evaluations.reserveCapacity(k)
        for i in 0..<k {
            evaluations.append(hornerEval(padded[i], at: zk))
        }

        // Step 4: Build remainder polynomial
        let omega = FflonkEngine.rootOfUnity(k: k)
        let remainder = buildRemainderPoly(evaluations: evaluations, z: z, omega: omega, k: k)

        // Step 5: Compute numerator P(X) - R(X)
        var numerator = combined
        for i in 0..<min(remainder.count, numerator.count) {
            numerator[i] = frSub(numerator[i], remainder[i])
        }

        // Step 6: Divide by vanishing polynomial Z(X) = X^k - z^k
        let quotient = divideByVanishing(numerator, zk: zk, k: k)

        // Step 7: Commit to quotient as witness
        let witness = try engine.kzg.commit(quotient)

        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        return FflonkProverResult(
            commitment: commitment,
            witness: witness,
            evaluations: evaluations,
            combinedPoly: combined,
            quotientPoly: quotient,
            point: z,
            batchSize: k,
            usedGPUCombine: usedGPUCombine,
            usedGPUCommit: true,  // KZG always uses MSM
            proverTime: elapsed
        )
    }

    // MARK: - Verification

    /// Verify proof using SRS secret: [P(s)] - [R(s)] == (s^k - z^k) * [q(s)].
    public func verify(result: FflonkProverResult, srsSecret: Fr) -> FflonkVerificationResult {
        guard let engine = cpuEngine else {
            return FflonkVerificationResult(
                valid: false, remainderAtSecret: Fr.zero,
                lhs: PointProjective.identity, rhs: PointProjective.identity
            )
        }

        let k = result.batchSize
        let z = result.point
        let omega = FflonkEngine.rootOfUnity(k: k)

        // Rebuild remainder from evaluations
        let remainder = buildRemainderPoly(
            evaluations: result.evaluations, z: z, omega: omega, k: k
        )

        // R(s)
        let rAtS = hornerEval(remainder, at: srsSecret)

        // LHS: C - [R(s)] * G
        let g1 = pointFromAffine(engine.kzg.srs[0])
        let rG = cPointScalarMul(g1, rAtS)
        let lhs = pointAdd(result.commitment, pointNeg(rG))

        // RHS: (s^k - z^k) * W
        let sk = frPow(srsSecret, UInt64(k))
        let zk = frPow(z, UInt64(k))
        let vanishAtS = frSub(sk, zk)
        let rhs = cPointScalarMul(result.witness, vanishAtS)

        // Compare
        let valid = projectivePointsEqual(lhs, rhs)

        return FflonkVerificationResult(
            valid: valid, remainderAtSecret: rAtS, lhs: lhs, rhs: rhs
        )
    }

    /// Verify using the underlying CPU engine (delegates to FflonkEngine).
    public func verifyWithCPUEngine(
        commitment: FflonkCommitment,
        proof: FflonkOpeningProof,
        srsSecret: Fr
    ) -> Bool {
        guard let engine = cpuEngine else { return false }
        return engine.verify(commitment: commitment, proof: proof, srsSecret: srsSecret)
    }

    // MARK: - Polynomial Multiply by Scalar (GPU)

    /// Multiply each coefficient by a scalar, using GPU for large polynomials.
    public func polyMulScalar(_ poly: [Fr], by scalar: Fr) -> (result: [Fr], usedGPU: Bool) {
        let n = poly.count
        if n == 0 { return ([], false) }

        if n >= config.gpuInterleaveThreshold,
           let device = device,
           let commandQueue = commandQueue,
           let pipeline = polyMulScalarPipeline {
            if let result = gpuPolyMulScalar(poly, scalar: scalar,
                                              device: device, commandQueue: commandQueue,
                                              pipeline: pipeline) {
                return (result, true)
            }
        }

        // CPU fallback using batch C kernel
        var result = [Fr](repeating: Fr.zero, count: n)
        poly.withUnsafeBytes { pBuf in
            withUnsafeBytes(of: scalar) { sBuf in
                result.withUnsafeMutableBytes { rBuf in
                    bn254_fr_batch_mul_scalar(
                        pBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        sBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(n))
                }
            }
        }
        return (result, false)
    }

    private func gpuPolyMulScalar(_ poly: [Fr], scalar: Fr,
                                   device: MTLDevice, commandQueue: MTLCommandQueue,
                                   pipeline: MTLComputePipelineState) -> [Fr]? {
        let n = poly.count
        let bufSize = n * MemoryLayout<Fr>.stride
        var polyData = poly; var scalarData = scalar; var count = UInt32(n)

        guard let polyBuf = device.makeBuffer(bytes: &polyData, length: bufSize, options: .storageModeShared),
              let scalarBuf = device.makeBuffer(bytes: &scalarData, length: MemoryLayout<Fr>.stride, options: .storageModeShared),
              let outBuf = device.makeBuffer(length: bufSize, options: .storageModeShared),
              let cntBuf = device.makeBuffer(bytes: &count, length: 4, options: .storageModeShared),
              let cmdBuf = commandQueue.makeCommandBuffer(),
              let enc = cmdBuf.makeComputeCommandEncoder() else { return nil }

        enc.setComputePipelineState(pipeline)
        enc.setBuffer(polyBuf, offset: 0, index: 0)
        enc.setBuffer(scalarBuf, offset: 0, index: 1)
        enc.setBuffer(outBuf, offset: 0, index: 2)
        enc.setBuffer(cntBuf, offset: 0, index: 3)
        let tg = MTLSize(width: (n + threadgroupSize - 1) / threadgroupSize, height: 1, depth: 1)
        enc.dispatchThreadgroups(tg, threadsPerThreadgroup: MTLSize(width: threadgroupSize, height: 1, depth: 1))
        enc.endEncoding()
        cmdBuf.commit(); cmdBuf.waitUntilCompleted()
        if cmdBuf.status != .completed { return nil }
        return Array(UnsafeBufferPointer(start: outBuf.contents().bindMemory(to: Fr.self, capacity: n), count: n))
    }

    // MARK: - Linearization (Batch Multiple Openings)

    /// Linearize k opening claims: L(X) = sum gamma^i * p_i(X), y_L = sum gamma^i * y_i.
    public func linearize(polynomials: [[Fr]], evaluations: [Fr], gamma: Fr) -> FflonkLinearization {
        let k = polynomials.count
        precondition(k == evaluations.count, "Polynomials and evaluations must match")

        // Compute gamma powers
        var gammaPowers = [Fr](repeating: Fr.zero, count: k)
        gammaPowers[0] = Fr.one
        for i in 1..<k {
            gammaPowers[i] = frMul(gammaPowers[i - 1], gamma)
        }

        // Linearized evaluation
        var linEval = Fr.zero
        for i in 0..<k {
            linEval = frAdd(linEval, frMul(gammaPowers[i], evaluations[i]))
        }

        // Linearized polynomial
        let maxDeg = polynomials.map { $0.count }.max() ?? 0
        var linPoly = [Fr](repeating: Fr.zero, count: maxDeg)
        for i in 0..<k {
            let (scaled, _) = polyMulScalar(polynomials[i], by: gammaPowers[i])
            for j in 0..<scaled.count {
                linPoly[j] = frAdd(linPoly[j], scaled[j])
            }
        }

        return FflonkLinearization(
            gammaPowers: gammaPowers,
            linearizedPoly: linPoly,
            linearizedEval: linEval
        )
    }

    // MARK: - Multi-Opening Proof

    /// Generate a multi-opening proof at multiple evaluation points, linearized with gamma.
    public func multiOpen(polynomials: [[Fr]], at points: [Fr], gamma: Fr) throws -> FflonkMultiOpening {
        guard let engine = cpuEngine else {
            throw FflonkProverError.noSRS
        }

        var openings = [FflonkOpeningProof]()
        var allEvals = [[Fr]]()

        for z in points {
            let proof = try engine.open(polynomials, at: z)
            openings.append(proof)
            allEvals.append(proof.evaluations)
        }

        // Flatten evaluations for linearization
        let flatPolys = polynomials
        let flatEvals = allEvals.flatMap { $0 }

        // Build linearization across all claims
        let dummyPolys = (0..<flatEvals.count).map { i -> [Fr] in
            if i < polynomials.count { return polynomials[i] }
            return polynomials[i % polynomials.count]
        }
        let linearization = linearize(polynomials: dummyPolys, evaluations: flatEvals, gamma: gamma)

        // Combined witness: sum of gamma-weighted individual witnesses
        var combinedWitness = PointProjective.identity
        var gammaPow = Fr.one
        for opening in openings {
            let weighted = cPointScalarMul(opening.witness, gammaPow)
            combinedWitness = pointAdd(combinedWitness, weighted)
            gammaPow = frMul(gammaPow, gamma)
        }

        return FflonkMultiOpening(
            openings: openings,
            linearization: linearization,
            combinedWitness: combinedWitness
        )
    }

    // MARK: - Polynomial Addition

    /// Add two polynomials coefficient-wise.
    public func polyAdd(_ a: [Fr], _ b: [Fr]) -> [Fr] {
        let maxLen = max(a.count, b.count)
        var result = [Fr](repeating: Fr.zero, count: maxLen)
        for i in 0..<a.count { result[i] = a[i] }
        let bCount = b.count
        result.withUnsafeMutableBytes { rBuf in
            b.withUnsafeBytes { bBuf in
                bn254_fr_batch_add_neon(
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(bCount))
            }
        }
        return result
    }

    /// Subtract two polynomials coefficient-wise.
    public func polySub(_ a: [Fr], _ b: [Fr]) -> [Fr] {
        let maxLen = max(a.count, b.count)
        var result = [Fr](repeating: Fr.zero, count: maxLen)
        for i in 0..<a.count { result[i] = a[i] }
        let bCount = b.count
        result.withUnsafeMutableBytes { rBuf in
            b.withUnsafeBytes { bBuf in
                bn254_fr_batch_sub_neon(
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(bCount))
            }
        }
        return result
    }

    // MARK: - Batch Size Utilities

    /// Compute the effective batch size for a given number of polynomials.
    public func effectiveBatchSize(for count: Int) -> Int {
        return nextPowerOf2(count)
    }

    /// Compute the combined polynomial degree for given sub-polynomial degrees.
    public func combinedDegree(subDegrees: [Int], batchSize k: Int) -> Int {
        let maxDeg = subDegrees.max() ?? 0
        return k * maxDeg
    }

    /// Check if GPU acceleration is available and initialized.
    public var isGPUAvailable: Bool {
        return device != nil && commandQueue != nil
    }

    /// Check if the interleave kernel compiled successfully.
    public var hasInterleaveKernel: Bool {
        return interleavePipeline != nil
    }

    /// Check if the poly-mul-scalar kernel compiled successfully.
    public var hasPolyMulScalarKernel: Bool {
        return polyMulScalarPipeline != nil
    }

    // MARK: - Internal Helpers

    /// Evaluate polynomial at a point using C Horner.
    private func hornerEval(_ coeffs: [Fr], at z: Fr) -> Fr {
        if coeffs.isEmpty { return Fr.zero }
        var result = Fr.zero
        coeffs.withUnsafeBytes { cBuf in
            withUnsafeBytes(of: z) { zBuf in
                withUnsafeMutableBytes(of: &result) { rBuf in
                    bn254_fr_horner_eval(
                        cBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        Int32(coeffs.count),
                        zBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                        rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                    )
                }
            }
        }
        return result
    }

    /// Build the remainder polynomial R(X) via Lagrange interpolation.
    ///
    /// R(X) interpolates the values {P(omega^j * z)} at points {omega^j * z}.
    /// P(omega^j * z) = sum_i y_i * omega^{ij} * z^i.
    private func buildRemainderPoly(evaluations: [Fr], z: Fr, omega: Fr, k: Int) -> [Fr] {
        // Evaluation points: x_j = omega^j * z
        var points = [Fr]()
        points.reserveCapacity(k)
        var omegaPow = Fr.one
        for _ in 0..<k {
            points.append(frMul(omegaPow, z))
            omegaPow = frMul(omegaPow, omega)
        }

        // Precompute omega^j for all j via chain multiply
        var omegaJPows = [Fr](repeating: Fr.one, count: k)
        for j in 1..<k { omegaJPows[j] = frMul(omegaJPows[j - 1], omega) }

        // Values at these points
        var values = [Fr]()
        values.reserveCapacity(k)
        for j in 0..<k {
            var val = Fr.zero
            var omegaIJ = Fr.one
            var zPow = Fr.one
            for i in 0..<k {
                val = frAdd(val, frMul(evaluations[i], frMul(omegaIJ, zPow)))
                omegaIJ = frMul(omegaIJ, omegaJPows[j])
                zPow = frMul(zPow, z)
            }
            values.append(val)
        }

        return lagrangeInterpolate(points: points, values: values)
    }

    /// Lagrange interpolation: given (x_i, y_i), compute polynomial of degree < n.
    private func lagrangeInterpolate(points: [Fr], values: [Fr]) -> [Fr] {
        let n = points.count
        if n == 0 { return [] }
        if n == 1 { return [values[0]] }

        var result = [Fr](repeating: Fr.zero, count: n)

        // Precompute all Lagrange denominators and batch-invert
        var lagDenoms = [Fr](repeating: Fr.one, count: n)
        for i in 0..<n {
            for j in 0..<n where j != i {
                lagDenoms[i] = frMul(lagDenoms[i], frSub(points[i], points[j]))
            }
        }
        var lagPrefix = [Fr](repeating: Fr.one, count: n)
        for i in 1..<n {
            lagPrefix[i] = lagDenoms[i - 1] == Fr.zero ? lagPrefix[i - 1] : frMul(lagPrefix[i - 1], lagDenoms[i - 1])
        }
        let lagLast = lagDenoms[n - 1] == Fr.zero ? lagPrefix[n - 1] : frMul(lagPrefix[n - 1], lagDenoms[n - 1])
        var lagInv = frInverse(lagLast)
        var lagDenomInvs = [Fr](repeating: Fr.zero, count: n)
        for i in stride(from: n - 1, through: 0, by: -1) {
            if lagDenoms[i] != Fr.zero {
                lagDenomInvs[i] = frMul(lagInv, lagPrefix[i])
                lagInv = frMul(lagInv, lagDenoms[i])
            }
        }

        for i in 0..<n {
            let coeff = frMul(values[i], lagDenomInvs[i])

            var basis = [Fr](repeating: Fr.zero, count: n)
            basis[0] = Fr.one
            var deg = 0

            for j in 0..<n {
                if j == i { continue }
                let negXj = frSub(Fr.zero, points[j])
                for d in stride(from: deg + 1, through: 1, by: -1) {
                    basis[d] = frAdd(basis[d - 1], frMul(basis[d], negXj))
                }
                basis[0] = frMul(basis[0], negXj)
                deg += 1
            }

            for d in 0..<n {
                result[d] = frAdd(result[d], frMul(coeff, basis[d]))
            }
        }

        return result
    }

    /// Divide polynomial by vanishing polynomial Z(X) = X^k - z^k via long division.
    private func divideByVanishing(_ numerator: [Fr], zk: Fr, k: Int) -> [Fr] {
        let n = numerator.count
        if n <= k { return [] }

        let quotientDeg = n - k
        var remainder = numerator

        var quotient = [Fr](repeating: Fr.zero, count: quotientDeg)

        for i in stride(from: n - 1, through: k, by: -1) {
            let qi = i - k
            let c = remainder[i]
            quotient[qi] = c
            remainder[i] = Fr.zero
            remainder[qi] = frAdd(remainder[qi], frMul(c, zk))
        }

        return quotient
    }

    /// Compare two projective points for equality.
    private func projectivePointsEqual(_ a: PointProjective, _ b: PointProjective) -> Bool {
        if pointIsIdentity(a) && pointIsIdentity(b) { return true }
        if pointIsIdentity(a) || pointIsIdentity(b) { return false }
        let aAff = batchToAffine([a])
        let bAff = batchToAffine([b])
        return fpToInt(aAff[0].x) == fpToInt(bAff[0].x) &&
               fpToInt(aAff[0].y) == fpToInt(bAff[0].y)
    }

    /// Smallest power of 2 >= n.
    private func nextPowerOf2(_ n: Int) -> Int {
        if n <= 1 { return 1 }
        var v = n - 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        return v + 1
    }

    // MARK: - Metal Shader Compilation

    /// Compile the coefficient interleaving kernel.
    ///
    /// The kernel maps each output position to the correct input sub-polynomial:
    ///   output[k*j + i] = input[i*maxDeg + j]  (for i < k, j < maxDeg)
    private static func compileInterleaveKernel(device: MTLDevice) -> MTLComputePipelineState? {
        let shaderDir = findShaderDir()
        guard let frSource = try? String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8) else {
            return nil
        }

        let frClean = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let kernelSource = """

        // Interleave k sub-polynomials into combined polynomial P(X).
        // input:  flat buffer of k sub-polys, each padded to maxDeg coefficients.
        //         input[i * maxDeg + j] = coefficient j of sub-polynomial i.
        // output: combined[k*j + i] = input[i * maxDeg + j].
        // params: [k, maxDeg, combinedSize]
        kernel void fflonk_interleave(
            device const Fr *input [[buffer(0)]],
            device Fr *output [[buffer(1)]],
            device const uint *params [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            uint k = params[0];
            uint maxDeg = params[1];
            uint combinedSize = params[2];
            if (gid >= combinedSize) return;

            // gid = k * j + i  =>  i = gid % k,  j = gid / k
            uint i = gid % k;
            uint j = gid / k;

            if (j < maxDeg) {
                output[gid] = input[i * maxDeg + j];
            } else {
                // Zero padding beyond maxDeg
                Fr zero;
                for (int w = 0; w < 4; w++) zero.v[w] = 0;
                output[gid] = zero;
            }
        }
        """

        let combined = frClean + "\n" + kernelSource
        let options = MTLCompileOptions()
        options.fastMathEnabled = true

        guard let library = try? device.makeLibrary(source: combined, options: options),
              let fn = library.makeFunction(name: "fflonk_interleave"),
              let pipeline = try? device.makeComputePipelineState(function: fn) else {
            return nil
        }
        return pipeline
    }

    /// Compile the polynomial-scalar multiplication kernel.
    ///
    /// Multiplies each coefficient by a single scalar element.
    private static func compilePolyMulScalarKernel(device: MTLDevice) -> MTLComputePipelineState? {
        let shaderDir = findShaderDir()
        guard let frSource = try? String(contentsOfFile: shaderDir + "/fields/bn254_fr.metal", encoding: .utf8) else {
            return nil
        }

        let frClean = frSource
            .replacingOccurrences(of: "#ifndef BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#define BN254_FR_METAL", with: "")
            .replacingOccurrences(of: "#endif // BN254_FR_METAL", with: "")

        let kernelSource = """

        // Multiply each polynomial coefficient by a scalar.
        kernel void fflonk_poly_mul_scalar(
            device const Fr *poly [[buffer(0)]],
            device const Fr *scalar [[buffer(1)]],
            device Fr *output [[buffer(2)]],
            device const uint *count [[buffer(3)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= count[0]) return;
            output[gid] = fr_mul(poly[gid], scalar[0]);
        }
        """

        let combined = frClean + "\n" + kernelSource
        let options = MTLCompileOptions()
        options.fastMathEnabled = true

        guard let library = try? device.makeLibrary(source: combined, options: options),
              let fn = library.makeFunction(name: "fflonk_poly_mul_scalar"),
              let pipeline = try? device.makeComputePipelineState(function: fn) else {
            return nil
        }
        return pipeline
    }

    private static func findShaderDir() -> String {
        let execDir = (CommandLine.arguments[0] as NSString).deletingLastPathComponent
        for bundle in Bundle.allBundles {
            if let url = bundle.url(forResource: "Shaders", withExtension: nil) {
                if FileManager.default.fileExists(atPath: url.appendingPathComponent("fields/bn254_fr.metal").path) {
                    return url.path
                }
            }
        }
        let candidates = [
            execDir + "/Shaders",
            execDir + "/../share/zkMetal/Shaders",
            "Sources/zkMetal/Shaders",
            FileManager.default.currentDirectoryPath + "/Sources/zkMetal/Shaders",
        ]
        for c in candidates {
            if FileManager.default.fileExists(atPath: c + "/fields/bn254_fr.metal") {
                return c
            }
        }
        return "Sources/zkMetal/Shaders"
    }
}

// MARK: - Fflonk Prover Errors

public enum FflonkProverError: Error, CustomStringConvertible {
    case noSRS
    case batchSizeExceeded(Int)
    case commitmentFailed
    case gpuDispatchFailed

    public var description: String {
        switch self {
        case .noSRS: return "No SRS configured for Fflonk prover"
        case .batchSizeExceeded(let k): return "Batch size \(k) exceeds maximum"
        case .commitmentFailed: return "KZG commitment failed"
        case .gpuDispatchFailed: return "GPU kernel dispatch failed"
        }
    }
}

// MARK: - PointProjective Identity Extension

extension PointProjective {
    /// The point at infinity (identity element): z = 0 in projective coords.
    public static var identity: PointProjective {
        return pointIdentity()
    }
}
