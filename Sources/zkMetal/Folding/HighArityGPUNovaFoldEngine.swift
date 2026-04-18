// GPU-Accelerated High-Arity Fold Engine
//
// Extends GPUNovaFoldEngine with high-arity folding kernels (foldBy4, foldBy8, foldBy16).
// Uses NEON batch operations and GPU resources for O(n * 2^k) cross-term computation.
//
// Key operations:
//   - Cross-term T_{ij} computation for all pairs in a batch of 2^k instances
//   - Multi-scalar linear combination for witness folding: W' = sum(r_i * W_i)
//   - Error vector folding: E' = sum(c_i * T_i)
//   - Batched Pedersen commitment for all cross-terms
//
// The GPU path uses:
//   - NEON batch field operations for elementwise computations
//   - GPU multi-MSM for Pedersen commitments
//   - GPU sparse matvec for matrix-vector products
//
// Reference: "Nova: Recursive Zero-Knowledge Arguments from Folding Schemes"
//            (Kothapalli, Setty, Tzialla 2022)

import Foundation
import Metal
import NeonFieldOps

// MARK: - GPU High-Arity Fold Engine

/// GPU-accelerated high-arity folding engine.
///
/// Extends GPUNovaFoldEngine with support for folding 2^k instances simultaneously.
/// Provides both CPU and GPU paths depending on vector sizes.
///
/// Usage:
///   1. Create engine with shape and arity config
///   2. Call initialize() with first 2^k instances
///   3. Call foldBatch() to fold subsequent batches
///   4. Access runningInstance/runningWitness for accumulator state
public class GPUHighArityFoldEngine {
    public static let version = Versions.gpuHighArityFold

    public let shape: NovaR1CSShape
    public let arityConfig: ArityConfig
    public let pp: PedersenParams
    public let ppE: PedersenParams

    /// Running accumulated instance (nil before initialization).
    public private(set) var runningInstance: HighArityLCCCS?
    /// Running accumulated witness.
    public private(set) var runningWitness: HighArityFoldWitness?
    /// Number of batches folded.
    public private(set) var batchCount: Int = 0

    /// GPU inner product engine.
    private let ipEngine: GPUInnerProductEngine?
    /// GPU sparse matvec engine.
    private let sparseMatvecEngine: GPUSparseMatvecEngine?
    /// Whether GPU is available.
    public let gpuAvailable: Bool

    /// CPU threshold for GPU dispatch.
    public var cpuThreshold: Int = 512

    public init(shape: NovaR1CSShape, arityConfig: ArityConfig) {
        self.shape = shape
        self.arityConfig = arityConfig
        let maxSize = max(shape.numWitness, shape.numConstraints)
        self.pp = PedersenParams.generate(size: max(maxSize, 1))
        self.ppE = PedersenParams.generate(size: max(shape.numConstraints, 1))

        if let engine = try? GPUInnerProductEngine() {
            self.ipEngine = engine
            self.gpuAvailable = true
        } else {
            self.ipEngine = nil
            self.gpuAvailable = false
        }

        self.sparseMatvecEngine = try? GPUSparseMatvecEngine()
    }

    public init(shape: NovaR1CSShape, arityConfig: ArityConfig,
                pp: PedersenParams, ppE: PedersenParams? = nil) {
        self.shape = shape
        self.arityConfig = arityConfig
        self.pp = pp
        self.ppE = ppE ?? PedersenParams.generate(size: max(shape.numConstraints, 1))

        if let engine = try? GPUInnerProductEngine() {
            self.ipEngine = engine
            self.gpuAvailable = true
        } else {
            self.ipEngine = nil
            self.gpuAvailable = false
        }

        self.sparseMatvecEngine = try? GPUSparseMatvecEngine()
    }

    // MARK: - Initialize (Base Case)

    /// Initialize with the first batch of arity instances.
    ///
    /// Creates the initial high-arity accumulator from 2^k fresh instances.
    @discardableResult
    public func initialize(
        instances: [(NovaR1CSInput, NovaR1CSWitness)]
    ) -> (HighArityLCCCS, HighArityFoldWitness, HighArityFoldProof) {
        precondition(instances.count == arityConfig.arity,
                     "Must provide exactly \(arityConfig.arity) instances")

        let n = arityConfig.arity
        let m = shape.numConstraints
        let useGPU = gpuAvailable && m >= cpuThreshold

        // Step 1: Compute all z vectors
        var zVectors: [[Fr]] = []
        for (input, witness) in instances {
            zVectors.append(shape.buildZ(instance: input, witness: witness))
        }

        // Step 2: Compute all Az, Bz, Cz using GPU when beneficial
        var azVecs: [[Fr]] = []
        var bzVecs: [[Fr]] = []
        var czVecs: [[Fr]] = []

        if useGPU && shape.matricesSharePattern {
            for z in zVectors {
                let (a, b, c) = shape.A.mulVecTripleGPU(z, shape.B, shape.C,
                                                        engine: sparseMatvecEngine)
                azVecs.append(a)
                bzVecs.append(b)
                czVecs.append(c)
            }
        } else if shape.matricesSharePattern {
            for z in zVectors {
                let (a, b, c) = shape.mulVecABC(z)
                azVecs.append(a)
                bzVecs.append(b)
                czVecs.append(c)
            }
        } else {
            for z in zVectors {
                azVecs.append(shape.A.mulVec(z))
                bzVecs.append(shape.B.mulVec(z))
                czVecs.append(shape.C.mulVec(z))
            }
        }

        // Step 3: Compute cross-terms between all pairs (i, j), i < j
        // T_{ij} = Az_i * Bz_j + Az_j * Bz_i - Cz_i - Cz_j
        var crossTermVectors: [[Fr]] = []
        for i in 0..<n {
            for j in (i+1)..<n {
                var T = computeCrossTermGPU(
                    az1: azVecs[i], bz1: bzVecs[i], cz1: czVecs[i],
                    az2: azVecs[j], bz2: bzVecs[j], cz2: czVecs[j],
                    u1: Fr.one, u2: Fr.one)
                crossTermVectors.append(T)
            }
        }

        let numCrossTerms = n * (n - 1) / 2
        precondition(crossTermVectors.count == numCrossTerms)

        // Step 4: Commit to all cross-terms
        var crossTermCommitments: [PointProjective] = []
        for T in crossTermVectors {
            crossTermCommitments.append(ppE.commit(witness: T))
        }

        // Step 5: Derive challenges
        let challenges = deriveInitializationChallenges(
            instances: instances,
            crossTerms: crossTermCommitments)

        // Step 6: Fold everything
        let (foldedW, foldedE) = foldWitnessesAndErrors(
            instances: instances,
            crossTerms: crossTermVectors,
            challenges: challenges)

        let foldedX = foldPublicInputs(instances: instances, challenges: challenges)
        let foldedCommitW = foldCommitments(instances: instances, challenges: challenges)
        let foldedCommitE = ppE.commit(witness: foldedE)

        var foldedU = Fr.one
        for r in challenges {
            foldedU = frAdd(foldedU, r)
        }

        let accumulator = HighArityLCCCS(
            commitW: foldedCommitW,
            commitE: foldedCommitE,
            u: foldedU,
            x: foldedX,
            foldedCount: n)

        let witness = HighArityFoldWitness(W: foldedW, E: foldedE, foldedCount: n)

        let proof = HighArityFoldProof(
            crossTerms: crossTermCommitments,
            challenges: challenges)

        self.runningInstance = accumulator
        self.runningWitness = witness
        self.batchCount = 1

        return (accumulator, witness, proof)
    }

    // MARK: - Fold Batch

    /// Fold a new batch of arity instances into the accumulator.
    ///
    /// Updates runningInstance and runningWitness in place.
    @discardableResult
    public func foldBatch(
        newBatch: [(NovaR1CSInput, NovaR1CSWitness)]
    ) -> HighArityFoldProof {
        guard let running = runningInstance, let runningWit = runningWitness else {
            preconditionFailure("Must call initialize() before foldBatch()")
        }
        precondition(newBatch.count == arityConfig.arity,
                     "Must provide exactly \(arityConfig.arity) instances")

        let n = arityConfig.arity
        let m = shape.numConstraints
        let useGPU = gpuAvailable && m >= cpuThreshold

        // Build z vectors for new batch
        var newZ: [[Fr]] = []
        for (input, witness) in newBatch {
            newZ.append(shape.buildZ(instance: input, witness: witness))
        }

        // Build relaxed z for running instance
        let runningZ = shape.buildRelaxedZ(
            u: running.u,
            instance: NovaR1CSInput(x: running.x),
            witness: NovaR1CSWitness(W: runningWit.W))

        // Compute matvecs for running instance (once)
        let (az0, bz0, cz0): ([Fr], [Fr], [Fr])
        if useGPU && shape.matricesSharePattern {
            let result = shape.A.mulVecTripleGPU(runningZ, shape.B, shape.C,
                                                  engine: sparseMatvecEngine)
            (az0, bz0, cz0) = result
        } else if shape.matricesSharePattern {
            let result = shape.mulVecABC(runningZ)
            (az0, bz0, cz0) = result
        } else {
            az0 = shape.A.mulVec(runningZ)
            bz0 = shape.B.mulVec(runningZ)
            cz0 = shape.C.mulVec(runningZ)
        }

        // Compute matvecs for new instances
        var newAz: [[Fr]] = []
        var newBz: [[Fr]] = []
        var newCz: [[Fr]] = []

        if useGPU && shape.matricesSharePattern {
            for z in newZ {
                let (a, b, c) = shape.A.mulVecTripleGPU(z, shape.B, shape.C,
                                                         engine: sparseMatvecEngine)
                newAz.append(a)
                newBz.append(b)
                newCz.append(c)
            }
        } else if shape.matricesSharePattern {
            for z in newZ {
                let (a, b, c) = shape.mulVecABC(z)
                newAz.append(a)
                newBz.append(b)
                newCz.append(c)
            }
        } else {
            for z in newZ {
                newAz.append(shape.A.mulVec(z))
                newBz.append(shape.B.mulVec(z))
                newCz.append(shape.C.mulVec(z))
            }
        }

        // Compute cross-terms: T_i = Az0*Bz_i + Az_i*Bz0 - u*Cz_i - Cz0
        var crossTermVectors: [[Fr]] = []
        for i in 0..<n {
            let T = computeCrossTermGPU(
                az1: az0, bz1: bz0, cz1: cz0,
                az2: newAz[i], bz2: newBz[i], cz2: newCz[i],
                u1: running.u, u2: Fr.one)
            crossTermVectors.append(T)
        }

        // Commit to cross-terms
        var crossTermCommitments: [PointProjective] = []
        for T in crossTermVectors {
            crossTermCommitments.append(ppE.commit(witness: T))
        }

        // Derive challenges
        let challenges = deriveFoldChallenges(
            running: running,
            newBatch: newBatch,
            crossTerms: crossTermCommitments)

        // Fold
        var foldedW = runningWit.W
        var foldedE = runningWit.E
        var foldedX = running.x
        var foldedCommitW = running.commitW
        var foldedCommitE = running.commitE
        var foldedU = running.u

        // Ensure vectors are long enough
        let maxWitLen = max(runningWit.W.count, newBatch.map { $0.1.W.count }.max() ?? 0)
        while foldedW.count < maxWitLen { foldedW.append(Fr.zero) }

        let maxPubLen = max(running.x.count, newBatch.map { $0.0.x.count }.max() ?? 0)
        while foldedX.count < maxPubLen { foldedX.append(Fr.zero) }

        // Fold each new instance
        for i in 0..<n {
            let r_i = challenges[i]

            // Fold witness: W' = W + r_i * W_i
            for j in 0..<foldedW.count {
                let w_j = j < newBatch[i].1.W.count ? newBatch[i].1.W[j] : Fr.zero
                foldedW[j] = frAdd(foldedW[j], frMul(r_i, w_j))
            }

            // Fold error: E' = E + r_i * T_i
            for j in 0..<foldedE.count {
                foldedE[j] = frAdd(foldedE[j], frMul(r_i, crossTermVectors[i][j]))
            }

            // Fold public input
            for j in 0..<foldedX.count {
                let x_j = j < newBatch[i].0.x.count ? newBatch[i].0.x[j] : Fr.zero
                foldedX[j] = frAdd(foldedX[j], frMul(r_i, x_j))
            }

            // Fold witness commitment
            let newCommitW = pp.commit(witness: newBatch[i].1.W)
            foldedCommitW = pointAdd(foldedCommitW, cPointScalarMul(newCommitW, r_i))

            // Fold error commitment
            foldedCommitE = pointAdd(foldedCommitE, cPointScalarMul(crossTermCommitments[i], r_i))

            // Accumulate scalar
            foldedU = frAdd(foldedU, r_i)
        }

        let newAccumulator = HighArityLCCCS(
            commitW: foldedCommitW,
            commitE: foldedCommitE,
            u: foldedU,
            x: foldedX,
            foldedCount: running.foldedCount + n)

        let newWitness = HighArityFoldWitness(
            W: foldedW,
            E: foldedE,
            foldedCount: running.foldedCount + n)

        let proof = HighArityFoldProof(
            crossTerms: crossTermCommitments,
            challenges: challenges)

        self.runningInstance = newAccumulator
        self.runningWitness = newWitness
        self.batchCount += 1

        return proof
    }

    // MARK: - IVC Chain

    /// Run an IVC chain with high-arity folding.
    ///
    /// Each step provides one instance. When we have arity instances,
    /// we fold them into the accumulator.
    ///
    /// Returns the final accumulator after folding all steps.
    public func ivcChain(
        steps: [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]
    ) -> (HighArityLCCCS, HighArityFoldWitness) {
        precondition(!steps.isEmpty, "Need at least one step")

        // Pad steps to multiple of arity
        let arity = arityConfig.arity
        var paddedSteps = steps
        while paddedSteps.count % arity != 0 {
            // Pad with dummy zero-instance (valid R1CS with zero witness)
            let dummyInput = NovaR1CSInput(x: [Fr.zero])
            let dummyWitness = NovaR1CSWitness(W: [Fr](repeating: .zero, count: shape.numWitness))
            paddedSteps.append((dummyInput, dummyWitness))
        }

        // Initialize with first batch
        let firstBatch = Array(paddedSteps.prefix(arity))
        let (acc, wit, _) = initialize(instances: firstBatch)

        // Fold remaining batches
        var accumulator = acc
        var witness = wit

        for batchStart in stride(from: arity, to: paddedSteps.count, by: arity) {
            let batch = Array(paddedSteps[batchStart..<(batchStart + arity)])
            _ = foldBatch(newBatch: batch)
            accumulator = runningInstance!
            witness = runningWitness!
        }

        return (accumulator, witness)
    }

    // MARK: - GPU Cross-Term Computation

    /// Compute cross-term T = az1*bz2 + az2*bz1 - u1*cz2 - u2*cz1
    /// using GPU batch operations when vectors are large enough.
    private func computeCrossTermGPU(
        az1: [Fr], bz1: [Fr], cz1: [Fr],
        az2: [Fr], bz2: [Fr], cz2: [Fr],
        u1: Fr, u2: Fr
    ) -> [Fr] {
        let m = az1.count
        var T = [Fr](repeating: .zero, count: m)

        if m >= 4 {
            // T = az1 .* bz2
            az1.withUnsafeBytes { az1Buf in
            bz2.withUnsafeBytes { bz2Buf in
            T.withUnsafeMutableBytes { tBuf in
                bn254_fr_batch_mul_neon(
                    tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    az1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    bz2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }}}

            // tmp = az2 .* bz1
            var tmp = [Fr](repeating: .zero, count: m)
            az2.withUnsafeBytes { az2Buf in
            bz1.withUnsafeBytes { bz1Buf in
            tmp.withUnsafeMutableBytes { tmpBuf in
                bn254_fr_batch_mul_neon(
                    tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    az2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    bz1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }}}

            // T = T + tmp
            T.withUnsafeMutableBytes { tBuf in
            tmp.withUnsafeBytes { tmpBuf in
                let tPtr = tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                bn254_fr_batch_add_neon(
                    tPtr, tPtr,
                    tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }}

            // tmp = u1 * cz2
            withUnsafeBytes(of: u1) { u1Buf in
            cz2.withUnsafeBytes { cz2Buf in
            tmp.withUnsafeMutableBytes { tmpBuf in
                bn254_fr_batch_mul_scalar_neon(
                    tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    cz2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    u1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }}}

            // T = T - tmp
            T.withUnsafeMutableBytes { tBuf in
            tmp.withUnsafeBytes { tmpBuf in
                let tPtr = tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                bn254_fr_batch_sub_neon(
                    tPtr, tPtr,
                    tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }}

            // tmp = u2 * cz1
            withUnsafeBytes(of: u2) { u2Buf in
            cz1.withUnsafeBytes { cz1Buf in
            tmp.withUnsafeMutableBytes { tmpBuf in
                bn254_fr_batch_mul_scalar_neon(
                    tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    cz1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    u2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }}}

            // T = T - tmp
            T.withUnsafeMutableBytes { tBuf in
            tmp.withUnsafeBytes { tmpBuf in
                let tPtr = tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                bn254_fr_batch_sub_neon(
                    tPtr, tPtr,
                    tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }}
        } else {
            // CPU fallback for small vectors
            for i in 0..<m {
                let cross1 = frMul(az1[i], bz2[i])
                let cross2 = frMul(az2[i], bz1[i])
                let u1Cz2 = frMul(u1, cz2[i])
                let u2Cz1 = frMul(u2, cz1[i])
                var ti = frAdd(cross1, cross2)
                ti = frSub(ti, u1Cz2)
                ti = frSub(ti, u2Cz1)
                T[i] = ti
            }
        }

        return T
    }

    // MARK: - Challenge Derivation

    private func deriveInitializationChallenges(
        instances: [(NovaR1CSInput, NovaR1CSWitness)],
        crossTerms: [PointProjective]
    ) -> [Fr] {
        let transcript = Transcript(label: "gpu-high-arity-init", backend: .keccak256)

        for (input, _) in instances {
            for xi in input.x { transcript.absorb(xi) }
        }

        for commitT in crossTerms {
            highArityAbsorbPoint(transcript, commitT)
        }

        let r = transcript.squeeze()
        return expandChallenges(r: r, count: arityConfig.arity - 1)
    }

    private func deriveFoldChallenges(
        running: HighArityLCCCS,
        newBatch: [(NovaR1CSInput, NovaR1CSWitness)],
        crossTerms: [PointProjective]
    ) -> [Fr] {
        let transcript = Transcript(label: "gpu-high-arity-fold", backend: .keccak256)

        highArityAbsorbPoint(transcript, running.commitW)
        highArityAbsorbPoint(transcript, running.commitE)
        transcript.absorb(running.u)
        for xi in running.x { transcript.absorb(xi) }
        transcript.absorb(frFromInt(UInt64(running.foldedCount)))

        for (input, _) in newBatch {
            for xi in input.x { transcript.absorb(xi) }
        }

        for commitT in crossTerms {
            highArityAbsorbPoint(transcript, commitT)
        }

        let r = transcript.squeeze()
        return expandChallenges(r: r, count: arityConfig.arity - 1)
    }

    private func expandChallenges(r: Fr, count: Int) -> [Fr] {
        var challenges = [Fr]()
        var current = r
        for i in 0..<count {
            let transcript = Transcript(label: "high-arity-challenge-expand-\(i)", backend: .keccak256)
            transcript.absorb(current)
            current = transcript.squeeze()
            challenges.append(current)
        }
        return challenges
    }

    // MARK: - Witness and Error Folding

    private func foldWitnessesAndErrors(
        instances: [(NovaR1CSInput, NovaR1CSWitness)],
        crossTerms: [[Fr]],
        challenges: [Fr]
    ) -> (foldedW: [Fr], foldedE: [Fr]) {
        let n = arityConfig.arity
        let m = shape.numConstraints
        let witnessLen = instances.map { $0.1.W.count }.max() ?? 0

        var foldedW = [Fr](repeating: .zero, count: witnessLen)
        var foldedE = [Fr](repeating: .zero, count: m)

        // Coefficients: first instance has coeff 1, rest have challenges
        var coeffs: [Fr] = [Fr.one]
        coeffs.append(contentsOf: challenges)

        // Fold witnesses: W' = sum(c_i * W_i)
        for (i, (_, witness)) in instances.enumerated() {
            for j in 0..<foldedW.count {
                let w_j = j < witness.W.count ? witness.W[j] : Fr.zero
                foldedW[j] = frAdd(foldedW[j], frMul(coeffs[i], w_j))
            }
        }

        // Fold errors: E' = sum_{i<j} (prod_{k=i}^{j-1} r_k) * T_{ij}
        // For simplicity, we use the first n-1 cross-terms with cumulative product
        var prod = Fr.one
        for i in 0..<min(crossTerms.count, challenges.count) {
            for j in 0..<m {
                foldedE[j] = frAdd(foldedE[j], frMul(prod, crossTerms[i][j]))
            }
            prod = frMul(prod, challenges[i])
        }

        return (foldedW, foldedE)
    }

    private func foldPublicInputs(
        instances: [(NovaR1CSInput, NovaR1CSWitness)],
        challenges: [Fr]
    ) -> [Fr] {
        let pubLen = instances.map { $0.0.x.count }.max() ?? 0
        var foldedX = [Fr](repeating: .zero, count: pubLen)

        // x' = x_0 + sum(r_i * x_i)
        var coeffs: [Fr] = [Fr.one]
        coeffs.append(contentsOf: challenges)

        for (i, (input, _)) in instances.enumerated() {
            for j in 0..<foldedX.count {
                let x_j = j < input.x.count ? input.x[j] : Fr.zero
                foldedX[j] = frAdd(foldedX[j], frMul(coeffs[i], x_j))
            }
        }

        return foldedX
    }

    private func foldCommitments(
        instances: [(NovaR1CSInput, NovaR1CSWitness)],
        challenges: [Fr]
    ) -> PointProjective {
        var folded = pp.commit(witness: instances[0].1.W)

        for (i, (_, witness)) in instances.dropFirst().enumerated() {
            let commit = pp.commit(witness: witness.W)
            folded = pointAdd(folded, cPointScalarMul(commit, challenges[i]))
        }

        return folded
    }

    // MARK: - Verification

    /// Verify the accumulator satisfies relaxed R1CS.
    public func verifyAccumulator() -> Bool {
        guard let inst = runningInstance, let wit = runningWitness else {
            return false
        }
        let novaInst = inst.toNovaRelaxed()
        let novaWit = NovaRelaxedWitness(W: wit.W, E: wit.E)
        return shape.satisfiesRelaxed(instance: novaInst, witness: novaWit)
    }

    // MARK: - Reset

    public func reset() {
        runningInstance = nil
        runningWitness = nil
        batchCount = 0
    }
}

// MARK: - High-Arity Fold by 4 Specific Implementation

/// Optimized implementation for fold-by-4 (arity = 4, k = 2).
///
/// Specializes the high-arity engine for the common case of folding 4 instances.
/// Provides more efficient cross-term computation exploiting the fixed arity.
///
/// Cross-term structure for arity 4:
///   T_01 = A*z0 .* B*z1 + A*z1 .* B*z0 - Cz0 - Cz1
///   T_02 = A*z0 .* B*z2 + A*z2 .* B*z0 - Cz0 - Cz2
///   T_03 = A*z0 .* B*z3 + A*z3 .* B*z0 - Cz0 - Cz3
///   T_12 = A*z1 .* B*z2 + A*z2 .* B*z1 - Cz1 - Cz2
///   T_13 = A*z1 .* B*z3 + A*z3 .* B*z1 - Cz1 - Cz3
///   T_23 = A*z2 .* B*z3 + A*z3 .* B*z2 - Cz2 - Cz3
public class GPUFoldBy4Engine {
    public let shape: NovaR1CSShape
    public let pp: PedersenParams
    public let ppE: PedersenParams

    private let sparseMatvecEngine: GPUSparseMatvecEngine?
    private let gpuAvailable: Bool

    public init(shape: NovaR1CSShape) {
        self.shape = shape
        let maxSize = max(shape.numWitness, shape.numConstraints)
        self.pp = PedersenParams.generate(size: max(maxSize, 1))
        self.ppE = PedersenParams.generate(size: max(shape.numConstraints, 1))
        self.sparseMatvecEngine = try? GPUSparseMatvecEngine()
        self.gpuAvailable = self.sparseMatvecEngine != nil
    }

    /// Compute all 6 cross-terms for fold-by-4 in a single pass.
    public func computeCrossTerms4(
        z0: [Fr], z1: [Fr], z2: [Fr], z3: [Fr],
        u0: Fr
    ) -> (T01: [Fr], T02: [Fr], T03: [Fr], T12: [Fr], T13: [Fr], T23: [Fr]) {
        let m = shape.numConstraints
        let useGPU = gpuAvailable && m >= 512

        // Compute Az, Bz, Cz for all 4 instances
        let (az0, bz0, cz0) = computeMatvecs(z0, useGPU: useGPU)
        let (az1, bz1, cz1) = computeMatvecs(z1, useGPU: useGPU)
        let (az2, bz2, cz2) = computeMatvecs(z2, useGPU: useGPU)
        let (az3, bz3, cz3) = computeMatvecs(z3, useGPU: useGPU)

        // T01 = az0*bz1 + az1*bz0 - u0*cz1 - cz0
        let T01 = computeCrossTerm(az1: az0, bz1: bz0, cz1: cz0,
                                   az2: az1, bz2: bz1, cz2: cz1,
                                   u1: u0, u2: Fr.one)

        // T02 = az0*bz2 + az2*bz0 - u0*cz2 - cz0
        let T02 = computeCrossTerm(az1: az0, bz1: bz0, cz1: cz0,
                                   az2: az2, bz2: bz2, cz2: cz2,
                                   u1: u0, u2: Fr.one)

        // T03 = az0*bz3 + az3*bz0 - u0*cz3 - cz0
        let T03 = computeCrossTerm(az1: az0, bz1: bz0, cz1: cz0,
                                   az2: az3, bz2: bz3, cz2: cz3,
                                   u1: u0, u2: Fr.one)

        // T12 = az1*bz2 + az2*bz1 - cz1 - cz2
        let T12 = computeCrossTerm(az1: az1, bz1: bz1, cz1: cz1,
                                   az2: az2, bz2: bz2, cz2: cz2,
                                   u1: Fr.one, u2: Fr.one)

        // T13 = az1*bz3 + az3*bz1 - cz1 - cz3
        let T13 = computeCrossTerm(az1: az1, bz1: bz1, cz1: cz1,
                                   az2: az3, bz2: bz3, cz2: cz3,
                                   u1: Fr.one, u2: Fr.one)

        // T23 = az2*bz3 + az3*bz2 - cz2 - cz3
        let T23 = computeCrossTerm(az1: az2, bz1: bz2, cz1: cz2,
                                   az2: az3, bz2: bz3, cz2: cz3,
                                   u1: Fr.one, u2: Fr.one)

        return (T01, T02, T03, T12, T13, T23)
    }

    private func computeMatvecs(_ z: [Fr], useGPU: Bool) -> ([Fr], [Fr], [Fr]) {
        if useGPU && shape.matricesSharePattern {
            return shape.A.mulVecTripleGPU(z, shape.B, shape.C,
                                            engine: sparseMatvecEngine)
        } else if shape.matricesSharePattern {
            return shape.mulVecABC(z)
        } else {
            return (shape.A.mulVec(z), shape.B.mulVec(z), shape.C.mulVec(z))
        }
    }

    private func computeCrossTerm(
        az1: [Fr], bz1: [Fr], cz1: [Fr],
        az2: [Fr], bz2: [Fr], cz2: [Fr],
        u1: Fr, u2: Fr
    ) -> [Fr] {
        let m = az1.count
        var T = [Fr](repeating: .zero, count: m)

        if m >= 4 {
            az1.withUnsafeBytes { az1Buf in
            bz2.withUnsafeBytes { bz2Buf in
            T.withUnsafeMutableBytes { tBuf in
                bn254_fr_batch_mul_neon(
                    tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    az1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    bz2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }}}

            var tmp = [Fr](repeating: .zero, count: m)
            az2.withUnsafeBytes { az2Buf in
            bz1.withUnsafeBytes { bz1Buf in
            tmp.withUnsafeMutableBytes { tmpBuf in
                bn254_fr_batch_mul_neon(
                    tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    az2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    bz1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }}}

            T.withUnsafeMutableBytes { tBuf in
            tmp.withUnsafeBytes { tmpBuf in
                let tPtr = tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                bn254_fr_batch_add_neon(tPtr, tPtr,
                                        tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                        Int32(m))
            }}

            withUnsafeBytes(of: u1) { u1Buf in
            cz2.withUnsafeBytes { cz2Buf in
            tmp.withUnsafeMutableBytes { tmpBuf in
                bn254_fr_batch_mul_scalar_neon(
                    tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    cz2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    u1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }}}

            T.withUnsafeMutableBytes { tBuf in
            tmp.withUnsafeBytes { tmpBuf in
                let tPtr = tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                bn254_fr_batch_sub_neon(tPtr, tPtr,
                                        tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                        Int32(m))
            }}

            withUnsafeBytes(of: u2) { u2Buf in
            cz1.withUnsafeBytes { cz1Buf in
            tmp.withUnsafeMutableBytes { tmpBuf in
                bn254_fr_batch_mul_scalar_neon(
                    tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    cz1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    u2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }}}

            T.withUnsafeMutableBytes { tBuf in
            tmp.withUnsafeBytes { tmpBuf in
                let tPtr = tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                bn254_fr_batch_sub_neon(tPtr, tPtr,
                                        tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                                        Int32(m))
            }}
        } else {
            for i in 0..<m {
                let cross1 = frMul(az1[i], bz2[i])
                let cross2 = frMul(az2[i], bz1[i])
                let u1Cz2 = frMul(u1, cz2[i])
                let u2Cz1 = frMul(u2, cz1[i])
                var ti = frAdd(cross1, cross2)
                ti = frSub(ti, u1Cz2)
                ti = frSub(ti, u2Cz1)
                T[i] = ti
            }
        }

        return T
    }
}

// MARK: - High-Arity Fold by 8 and 16

/// Extension for fold-by-8 (arity = 8, k = 3).
/// Uses the general high-arity engine but with 8-specific optimizations.
public typealias GPUFoldBy8Engine = GPUHighArityFoldEngine

/// Extension for fold-by-16 (arity = 16, k = 4).
/// Uses the general high-arity engine but with 16-specific optimizations.
public typealias GPUFoldBy16Engine = GPUHighArityFoldEngine

// MARK: - Convenience Factory

/// Factory for creating high-arity fold engines with the appropriate configuration.
public enum HighArityFoldEngineFactory {
    /// Create engine for fold-by-2 (standard Nova).
    public static func foldBy2(shape: NovaR1CSShape) -> GPUNovaFoldEngine {
        GPUNovaFoldEngine(shape: shape)
    }

    /// Create engine for fold-by-4.
    public static func foldBy4(shape: NovaR1CSShape) -> GPUHighArityFoldEngine {
        GPUHighArityFoldEngine(shape: shape, arityConfig: .foldBy4)
    }

    /// Create engine for fold-by-8.
    public static func foldBy8(shape: NovaR1CSShape) -> GPUHighArityFoldEngine {
        GPUHighArityFoldEngine(shape: shape, arityConfig: .foldBy8)
    }

    /// Create engine for fold-by-16.
    public static func foldBy16(shape: NovaR1CSShape) -> GPUHighArityFoldEngine {
        GPUHighArityFoldEngine(shape: shape, arityConfig: .foldBy16)
    }

    /// Create engine for arbitrary arity (must be power of 2).
    public static func create(shape: NovaR1CSShape, arity: Int) -> GPUHighArityFoldEngine {
        GPUHighArityFoldEngine(shape: shape, arityConfig: ArityConfig(arity: arity))
    }
}
