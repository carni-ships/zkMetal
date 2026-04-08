// GPU-Accelerated Nova Folding Engine
//
// Implements GPU-accelerated Nova IVC folding for relaxed R1CS instances.
// The core operations (cross-term computation, witness linear combination,
// error vector folding) are dispatched to Metal GPU via the inner product
// and linear combination kernels.
//
// Architecture:
//   GPUNovaFoldEngine — top-level engine managing folding state
//     - computeCrossTermGPU: GPU-accelerated cross-term T computation
//     - foldWitnessGPU: GPU linear combination of witness vectors
//     - foldErrorGPU: GPU linear combination of error vectors
//     - Running instance accumulator for multi-step IVC
//
// The GPU acceleration targets the O(n) field operations in each fold:
//   - Cross-term T[i] = Az1[i]*Bz2[i] + Az2[i]*Bz1[i] - u1*Cz2[i] - Cz1[i]
//   - Witness fold W' = W1 + r * W2  (n field mul-adds)
//   - Error fold E' = E1 + r * T     (m field mul-adds)
//
// For commitment operations (Pedersen MSM), uses the existing MetalMSM engine.
//
// Reference: "Nova: Recursive Zero-Knowledge Arguments from Folding Schemes"
//            (Kothapalli, Setty, Tzialla 2022)

import Foundation
import Metal
import NeonFieldOps

// MARK: - GPU Nova Fold Engine

/// GPU-accelerated Nova folding engine for relaxed R1CS instance accumulation.
///
/// Usage:
///   1. Create engine with an R1CS shape
///   2. Call `initialize()` to set up the base instance
///   3. Call `foldStep()` repeatedly to fold new instances
///   4. Access `runningInstance` / `runningWitness` for the accumulator
///   5. Use `verify()` to check accumulator consistency
public class GPUNovaFoldEngine {
    public static let version = Versions.gpuNovaFold

    public let shape: NovaR1CSShape
    public let pp: PedersenParams
    public let ppE: PedersenParams

    /// Running accumulated relaxed instance (nil before initialization).
    public private(set) var runningInstance: NovaRelaxedInstance?
    /// Running accumulated relaxed witness.
    public private(set) var runningWitness: NovaRelaxedWitness?
    /// Number of IVC steps completed.
    public private(set) var stepCount: Int = 0

    /// GPU inner product engine for cross-term computation.
    private let ipEngine: GPUInnerProductEngine?
    /// Whether to use GPU (falls back to CPU if Metal unavailable).
    public let gpuAvailable: Bool

    /// Threshold: vectors shorter than this use CPU path.
    public var cpuThreshold: Int = 512

    /// Initialize with an R1CS shape. Generates Pedersen parameters and attempts
    /// to create GPU resources.
    public init(shape: NovaR1CSShape) {
        self.shape = shape
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
    }

    /// Initialize with pre-generated Pedersen parameters.
    public init(shape: NovaR1CSShape, pp: PedersenParams, ppE: PedersenParams? = nil) {
        self.shape = shape
        self.pp = pp
        self.ppE = ppE ?? PedersenParams.generate(size: max(shape.numConstraints, 1))

        if let engine = try? GPUInnerProductEngine() {
            self.ipEngine = engine
            self.gpuAvailable = true
        } else {
            self.ipEngine = nil
            self.gpuAvailable = false
        }
    }

    // MARK: - Initialize (base case)

    /// Initialize the IVC chain with the first computation step.
    /// Creates the initial relaxed R1CS instance with u=1, E=0.
    ///
    /// - Parameters:
    ///   - instance: public input for step 0
    ///   - witness: witness for step 0
    /// - Returns: the initial relaxed instance
    @discardableResult
    public func initialize(instance: NovaR1CSInput,
                           witness: NovaR1CSWitness) -> NovaRelaxedInstance {
        let (relaxedInst, relaxedWit) = shape.relax(instance: instance,
                                                      witness: witness,
                                                      pp: pp)
        self.runningInstance = relaxedInst
        self.runningWitness = relaxedWit
        self.stepCount = 1
        return relaxedInst
    }

    // MARK: - Cross-Term Computation (GPU-accelerated)

    /// Compute the cross-term vector T for folding two instances.
    ///
    /// T[i] = Az1[i]*Bz2[i] + Az2[i]*Bz1[i] - u1*Cz2[i] - Cz1[i]
    ///
    /// Uses GPU inner product engine for large vectors, CPU for small.
    public func computeCrossTerm(
        runningInstance: NovaRelaxedInstance,
        runningWitness: NovaRelaxedWitness,
        newInstance: NovaR1CSInput,
        newWitness: NovaR1CSWitness
    ) -> [Fr] {
        let z1 = shape.buildRelaxedZ(
            u: runningInstance.u,
            instance: NovaR1CSInput(x: runningInstance.x),
            witness: NovaR1CSWitness(W: runningWitness.W))
        let z2 = shape.buildZ(instance: newInstance, witness: newWitness)

        let az1 = shape.A.mulVec(z1)
        let bz1 = shape.B.mulVec(z1)
        let cz1 = shape.C.mulVec(z1)
        let az2 = shape.A.mulVec(z2)
        let bz2 = shape.B.mulVec(z2)
        let cz2 = shape.C.mulVec(z2)

        let m = shape.numConstraints
        var T = [Fr](repeating: .zero, count: m)

        if m >= 4 {
            let u1 = runningInstance.u
            // T = az1*bz2 + az2*bz1 - u1*cz2 - cz1
            // Step 1: T = az1 .* bz2
            az1.withUnsafeBytes { az1Buf in
            bz2.withUnsafeBytes { bz2Buf in
            T.withUnsafeMutableBytes { tBuf in
                bn254_fr_batch_mul_neon(
                    tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    az1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    bz2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }}}

            // Step 2: tmp = az2 .* bz1
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

            // Step 3: T = T + tmp
            T.withUnsafeMutableBytes { tBuf in
            tmp.withUnsafeBytes { tmpBuf in
                let tPtr = tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                bn254_fr_batch_add_neon(
                    tPtr,
                    tPtr,
                    tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }}

            // Step 4: tmp = u1 * cz2
            withUnsafeBytes(of: u1) { u1Buf in
            cz2.withUnsafeBytes { cz2Buf in
            tmp.withUnsafeMutableBytes { tmpBuf in
                bn254_fr_batch_mul_scalar_neon(
                    tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    cz2Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    u1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }}}

            // Step 5: T = T - tmp
            T.withUnsafeMutableBytes { tBuf in
            tmp.withUnsafeBytes { tmpBuf in
                let tPtr = tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                bn254_fr_batch_sub_neon(
                    tPtr,
                    tPtr,
                    tmpBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }}

            // Step 6: T = T - cz1
            T.withUnsafeMutableBytes { tBuf in
            cz1.withUnsafeBytes { cz1Buf in
                let tPtr = tBuf.baseAddress!.assumingMemoryBound(to: UInt64.self)
                bn254_fr_batch_sub_neon(
                    tPtr,
                    tPtr,
                    cz1Buf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(m))
            }}
        } else {
            let u1 = runningInstance.u
            for i in 0..<m {
                let cross1 = frMul(az1[i], bz2[i])
                let cross2 = frMul(az2[i], bz1[i])
                let uCz2 = frMul(u1, cz2[i])
                var ti = frAdd(cross1, cross2)
                ti = frSub(ti, uCz2)
                ti = frSub(ti, cz1[i])
                T[i] = ti
            }
        }
        return T
    }

    // MARK: - Linear Combination (GPU-accelerated)

    /// Compute the linear combination: result[i] = a[i] + r * b[i]
    /// Uses NEON-accelerated C function for large vectors.
    private func linearCombine(_ a: [Fr], _ b: [Fr], r: Fr) -> [Fr] {
        let n = a.count
        precondition(n == b.count)
        if n == 0 { return [] }

        var result = [Fr](repeating: .zero, count: n)

        // Use the NEON-accelerated bn254_fr_linear_combine for vectors
        if n >= 4 {
            a.withUnsafeBytes { aBuf in
            b.withUnsafeBytes { bBuf in
            withUnsafeBytes(of: r) { rBuf in
            result.withUnsafeMutableBytes { resBuf in
                bn254_fr_linear_combine(
                    aBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    bBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(n)
                )
            }}}}
        } else {
            for i in 0..<n {
                result[i] = frAdd(a[i], frMul(r, b[i]))
            }
        }
        return result
    }

    // MARK: - Fold Step

    /// Fold a fresh R1CS instance into the running relaxed instance.
    ///
    /// Returns the folding proof (commitment to cross-term T).
    /// Updates runningInstance and runningWitness in place.
    ///
    /// The folding is homomorphic:
    ///   commitW' = commitW1 + r * commitW2
    ///   commitE' = commitE1 + r * commitT
    ///   u' = u1 + r
    ///   x' = x1 + r * x2
    ///   W' = W1 + r * W2
    ///   E' = E1 + r * T
    @discardableResult
    public func foldStep(
        newInstance: NovaR1CSInput,
        newWitness: NovaR1CSWitness
    ) -> NovaFoldProof {
        guard let running = runningInstance, let runWit = runningWitness else {
            preconditionFailure("Must call initialize() before foldStep()")
        }

        // Step 1: Compute cross-term T
        let T = computeCrossTerm(
            runningInstance: running,
            runningWitness: runWit,
            newInstance: newInstance,
            newWitness: newWitness)

        // Step 2: Commit to T
        let commitT = ppE.commit(witness: T)

        // Step 3: Derive challenge r via Fiat-Shamir
        let r = deriveChallenge(
            runningInstance: running,
            newInstance: newInstance,
            commitT: commitT)

        // Step 4: Fold commitments
        let commitW2 = pp.commit(witness: newWitness.W)
        let foldedCommitW = pointAdd(running.commitW,
                                      cPointScalarMul(commitW2, r))
        let foldedCommitE = pointAdd(running.commitE,
                                      cPointScalarMul(commitT, r))

        // Step 5: Fold scalar u' = u1 + r
        let foldedU = frAdd(running.u, r)

        // Step 6: Fold public input x' = x1 + r * x2
        let foldedX = linearCombine(running.x, newInstance.x, r: r)

        // Step 7: Fold witness W' = W1 + r * W2
        let foldedW = linearCombine(runWit.W, newWitness.W, r: r)

        // Step 8: Fold error E' = E1 + r * T
        let foldedE = linearCombine(runWit.E, T, r: r)

        let foldedInst = NovaRelaxedInstance(
            commitW: foldedCommitW,
            commitE: foldedCommitE,
            u: foldedU,
            x: foldedX)
        let foldedWit = NovaRelaxedWitness(W: foldedW, E: foldedE)

        self.runningInstance = foldedInst
        self.runningWitness = foldedWit
        self.stepCount += 1

        return NovaFoldProof(commitT: commitT)
    }

    // MARK: - Fold (non-mutating variant)

    /// Fold without modifying engine state. Returns folded instance, witness, and proof.
    public func fold(
        runningInstance: NovaRelaxedInstance,
        runningWitness: NovaRelaxedWitness,
        newInstance: NovaR1CSInput,
        newWitness: NovaR1CSWitness
    ) -> (NovaRelaxedInstance, NovaRelaxedWitness, NovaFoldProof) {
        let T = computeCrossTerm(
            runningInstance: runningInstance,
            runningWitness: runningWitness,
            newInstance: newInstance,
            newWitness: newWitness)

        let commitT = ppE.commit(witness: T)

        let r = deriveChallenge(
            runningInstance: runningInstance,
            newInstance: newInstance,
            commitT: commitT)

        let commitW2 = pp.commit(witness: newWitness.W)
        let foldedCommitW = pointAdd(runningInstance.commitW,
                                      cPointScalarMul(commitW2, r))
        let foldedCommitE = pointAdd(runningInstance.commitE,
                                      cPointScalarMul(commitT, r))

        let foldedU = frAdd(runningInstance.u, r)
        let foldedX = linearCombine(runningInstance.x, newInstance.x, r: r)
        let foldedW = linearCombine(runningWitness.W, newWitness.W, r: r)
        let foldedE = linearCombine(runningWitness.E, T, r: r)

        let foldedInst = NovaRelaxedInstance(
            commitW: foldedCommitW,
            commitE: foldedCommitE,
            u: foldedU,
            x: foldedX)
        let foldedWit = NovaRelaxedWitness(W: foldedW, E: foldedE)

        return (foldedInst, foldedWit, NovaFoldProof(commitT: commitT))
    }

    // MARK: - IVC Chain

    /// Run an IVC chain: fold a sequence of (instance, witness) pairs.
    ///
    /// The first pair is relaxed to form the base case. Each subsequent pair
    /// is folded into the running accumulated instance.
    ///
    /// Returns the final accumulated relaxed instance and witness.
    public func ivcChain(
        steps: [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]
    ) -> (NovaRelaxedInstance, NovaRelaxedWitness) {
        precondition(!steps.isEmpty, "Need at least one step")

        initialize(instance: steps[0].instance, witness: steps[0].witness)

        for i in 1..<steps.count {
            foldStep(newInstance: steps[i].instance, newWitness: steps[i].witness)
        }

        return (runningInstance!, runningWitness!)
    }

    // MARK: - Accumulator Verification

    /// Verify that the running accumulator satisfies the relaxed R1CS relation.
    /// This is the "decider" check: Az . Bz = u*(Cz) + E.
    public func verifyAccumulator() -> Bool {
        guard let inst = runningInstance, let wit = runningWitness else {
            return false
        }
        return shape.satisfiesRelaxed(instance: inst, witness: wit)
    }

    /// Verify a specific relaxed instance/witness pair satisfies relaxed R1CS.
    public func verify(instance: NovaRelaxedInstance,
                       witness: NovaRelaxedWitness) -> Bool {
        return shape.satisfiesRelaxed(instance: instance, witness: witness)
    }

    // MARK: - Reset

    /// Reset the engine state, clearing the running accumulator.
    public func reset() {
        runningInstance = nil
        runningWitness = nil
        stepCount = 0
    }

    // MARK: - Derive Challenge (Fiat-Shamir)

    /// Derive the folding challenge r from the transcript.
    public func deriveChallenge(
        runningInstance: NovaRelaxedInstance,
        newInstance: NovaR1CSInput,
        commitT: PointProjective
    ) -> Fr {
        let transcript = Transcript(label: "gpu-nova-fold", backend: .keccak256)

        novaAbsorbPoint(transcript, runningInstance.commitW)
        novaAbsorbPoint(transcript, runningInstance.commitE)
        transcript.absorb(runningInstance.u)
        for xi in runningInstance.x { transcript.absorb(xi) }

        for xi in newInstance.x { transcript.absorb(xi) }

        novaAbsorbPoint(transcript, commitT)

        return transcript.squeeze()
    }

    // MARK: - GPU Inner Product for Cross-Term Components

    /// Compute elementwise inner product using GPU when beneficial.
    /// Returns Sigma a[i] * b[i].
    public func gpuFieldInnerProduct(_ a: [Fr], _ b: [Fr]) -> Fr {
        if let engine = ipEngine, a.count >= cpuThreshold {
            return engine.fieldInnerProduct(a: a, b: b)
        }
        // CPU fallback
        var acc = Fr.zero
        for i in 0..<a.count {
            acc = frAdd(acc, frMul(a[i], b[i]))
        }
        return acc
    }
}
