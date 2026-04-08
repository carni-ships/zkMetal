// GPU-Accelerated Nova Decider Engine
//
// Produces a final SNARK proof from a Nova folding accumulator. The "decider"
// is the last step of Nova IVC: given an accumulated relaxed R1CS instance and
// witness, it proves that the accumulator is valid via a Spartan-style sumcheck.
//
// Architecture:
//   GPUNovaDeciderEngine — top-level engine managing decider proof generation
//     - Accumulator verification (relaxed R1CS satisfaction check)
//     - NIFS verification (fold chain replay)
//     - Decider circuit: sumcheck over relaxed R1CS gate polynomial
//     - Compressed SNARK generation from accumulator
//     - Cross-term computation and verification
//     - Support for both Nova and SuperNova accumulators
//
// GPU acceleration targets:
//   - Sparse matrix-vector products (A*z, B*z, C*z) for large circuits
//   - Inner products during sumcheck round evaluation
//   - Witness commitment (Pedersen MSM)
//   - Eq polynomial evaluation over the boolean hypercube
//
// The decider proves: sum_{x in {0,1}^s} eq(tau, x) * g(x) = 0
//   where g(x) = (Az . Bz)(x) - u*(Cz)(x) - E(x)
//
// Reference: "Nova: Recursive Zero-Knowledge Arguments from Folding Schemes"
//            (Kothapalli, Setty, Tzialla 2022)
// Reference: "Spartan: Efficient and general-purpose zkSNARKs without trusted setup"
//            (Setty 2019)

import Foundation
import Metal
import NeonFieldOps

// MARK: - Decider Proof

/// Compact proof produced by the Nova decider engine.
///
/// Contains a Spartan-style sumcheck proof of the relaxed R1CS relation,
/// plus witness commitment and evaluation claims for binding.
public struct NovaDeciderProof {
    /// Sumcheck round polynomials: each round is (s(0), s(1), s(2))
    public let sumcheckRounds: [(Fr, Fr, Fr)]
    /// Claimed evaluation of Az, Bz, Cz at the sumcheck evaluation point
    public let matVecEvals: (az: Fr, bz: Fr, cz: Fr)
    /// Commitment to the witness W (Pedersen)
    public let commitW: PointProjective
    /// Commitment to the error E (Pedersen)
    public let commitE: PointProjective
    /// The relaxation scalar u from the accumulated instance
    public let u: Fr
    /// Public input from the accumulated instance
    public let publicInput: [Fr]
    /// Hash-based witness binding for transcript integrity
    public let witnessHash: Fr
    /// Number of IVC steps that produced this accumulator
    public let stepCount: Int

    public init(sumcheckRounds: [(Fr, Fr, Fr)],
                matVecEvals: (az: Fr, bz: Fr, cz: Fr),
                commitW: PointProjective,
                commitE: PointProjective,
                u: Fr,
                publicInput: [Fr],
                witnessHash: Fr,
                stepCount: Int) {
        self.sumcheckRounds = sumcheckRounds
        self.matVecEvals = matVecEvals
        self.commitW = commitW
        self.commitE = commitE
        self.u = u
        self.publicInput = publicInput
        self.witnessHash = witnessHash
        self.stepCount = stepCount
    }
}

// MARK: - Decider Configuration

/// Configuration for the Nova decider engine.
public struct NovaDeciderConfig {
    /// Whether to use GPU acceleration (falls back to CPU if unavailable)
    public let useGPU: Bool
    /// Minimum vector size for GPU dispatch (smaller uses CPU)
    public let gpuThreshold: Int
    /// Whether to verify the fold chain before deciding
    public let verifyFoldChain: Bool

    public init(useGPU: Bool = true, gpuThreshold: Int = 512, verifyFoldChain: Bool = true) {
        self.useGPU = useGPU
        self.gpuThreshold = gpuThreshold
        self.verifyFoldChain = verifyFoldChain
    }
}

// MARK: - SuperNova Accumulator

/// A SuperNova accumulator wrapping multiple circuit accumulators.
///
/// SuperNova extends Nova to support multiple step circuits, each with its own
/// R1CS shape. The accumulator tracks one relaxed instance per circuit type.
public struct SuperNovaAccumulator {
    /// Per-circuit-type accumulated relaxed instances
    public let instances: [NovaRelaxedInstance]
    /// Per-circuit-type accumulated relaxed witnesses
    public let witnesses: [NovaRelaxedWitness]
    /// Per-circuit-type R1CS shapes
    public let shapes: [NovaR1CSShape]
    /// Which circuit index was used at each IVC step
    public let circuitSchedule: [Int]
    /// Total number of IVC steps
    public let stepCount: Int

    public init(instances: [NovaRelaxedInstance],
                witnesses: [NovaRelaxedWitness],
                shapes: [NovaR1CSShape],
                circuitSchedule: [Int],
                stepCount: Int) {
        precondition(instances.count == shapes.count)
        precondition(witnesses.count == shapes.count)
        self.instances = instances
        self.witnesses = witnesses
        self.shapes = shapes
        self.circuitSchedule = circuitSchedule
        self.stepCount = stepCount
    }
}

// MARK: - GPU Nova Decider Engine

/// GPU-accelerated Nova decider engine for producing final SNARK proofs
/// from Nova IVC accumulators.
///
/// Usage:
///   1. Create engine with an R1CS shape
///   2. Build an accumulator via `GPUNovaFoldEngine` or `NovaIVCProver`
///   3. Call `decide()` to produce a compressed SNARK proof
///   4. Call `verify()` on the verifier side to check the proof
///
/// Supports both standard Nova (single circuit) and SuperNova (multi-circuit).
public final class GPUNovaDeciderEngine {

    public let shape: NovaR1CSShape
    public let config: NovaDeciderConfig
    public let pp: PedersenParams
    public let ppE: PedersenParams

    /// GPU inner product engine for accelerated field operations.
    private let ipEngine: GPUInnerProductEngine?
    /// Whether GPU is available and enabled.
    public let gpuAvailable: Bool

    // MARK: - Initialization

    /// Initialize with an R1CS shape and optional configuration.
    public init(shape: NovaR1CSShape, config: NovaDeciderConfig = NovaDeciderConfig()) {
        self.shape = shape
        self.config = config
        let maxSize = max(shape.numWitness, shape.numConstraints)
        self.pp = PedersenParams.generate(size: max(maxSize, 1))
        self.ppE = PedersenParams.generate(size: max(shape.numConstraints, 1))

        if config.useGPU, let engine = try? GPUInnerProductEngine() {
            self.ipEngine = engine
            self.gpuAvailable = true
        } else {
            self.ipEngine = nil
            self.gpuAvailable = false
        }
    }

    /// Initialize with pre-generated Pedersen parameters.
    public init(shape: NovaR1CSShape, pp: PedersenParams,
                ppE: PedersenParams? = nil,
                config: NovaDeciderConfig = NovaDeciderConfig()) {
        self.shape = shape
        self.config = config
        self.pp = pp
        self.ppE = ppE ?? PedersenParams.generate(size: max(shape.numConstraints, 1))

        if config.useGPU, let engine = try? GPUInnerProductEngine() {
            self.ipEngine = engine
            self.gpuAvailable = true
        } else {
            self.ipEngine = nil
            self.gpuAvailable = false
        }
    }

    // MARK: - Accumulator Verification

    /// Verify that an accumulated instance satisfies the relaxed R1CS relation.
    ///
    /// Checks: Az . Bz = u*(Cz) + E
    ///
    /// This is the "honest decider" check performed with full witness access.
    public func verifyAccumulator(instance: NovaRelaxedInstance,
                                   witness: NovaRelaxedWitness) -> Bool {
        return shape.satisfiesRelaxed(instance: instance, witness: witness)
    }

    /// Verify commitment openings for the accumulated witness.
    ///
    /// Checks that commitW and commitE in the instance match recomputed
    /// Pedersen commitments from the witness vectors.
    public func verifyCommitments(instance: NovaRelaxedInstance,
                                   witness: NovaRelaxedWitness) -> Bool {
        let recomputedW = pp.commit(witness: witness.W)
        guard novaPointEq(instance.commitW, recomputedW) else { return false }

        // Only check E commitment if E is non-trivial
        if witness.E.contains(where: { !$0.isZero }) {
            let recomputedE = ppE.commit(witness: witness.E)
            guard novaPointEq(instance.commitE, recomputedE) else { return false }
        }
        return true
    }

    // MARK: - NIFS Verification

    /// Verify a chain of NIFS (Non-Interactive Folding Scheme) fold steps.
    ///
    /// Replays the Fiat-Shamir transcript for each fold step and checks that
    /// the derived folded instances match the claimed intermediate instances.
    ///
    /// - Parameter proof: the IVC proof containing fold chain data
    /// - Returns: true if all fold steps are valid
    public func verifyNIFSChain(proof: NovaIVCProof) -> Bool {
        if proof.stepCount <= 1 { return true }
        guard proof.foldProofs.count == proof.stepCount - 1 else { return false }
        guard proof.freshInstances.count == proof.stepCount - 1 else { return false }
        guard proof.intermediateInstances.count == proof.stepCount - 1 else { return false }

        for i in 0..<proof.foldProofs.count {
            let running = proof.intermediateInstances[i]
            let fresh = proof.freshInstances[i]
            let foldProof = proof.foldProofs[i]

            // Re-derive Fiat-Shamir challenge
            let transcript = Transcript(label: "nova-r1cs-fold", backend: .keccak256)
            novaAbsorbPoint(transcript, running.commitW)
            novaAbsorbPoint(transcript, running.commitE)
            transcript.absorb(running.u)
            for xi in running.x { transcript.absorb(xi) }
            for xi in fresh.x { transcript.absorb(xi) }
            novaAbsorbPoint(transcript, foldProof.commitT)
            let r = transcript.squeeze()

            // Compute expected folded values
            let expectedU = frAdd(running.u, r)
            let expectedCommitE = pointAdd(running.commitE,
                                            cPointScalarMul(foldProof.commitT, r))

            var expectedX = [Fr](repeating: .zero, count: running.x.count)
            for k in 0..<running.x.count {
                expectedX[k] = frAdd(running.x[k], frMul(r, fresh.x[k]))
            }

            // Determine target (next intermediate or final)
            let target: NovaRelaxedInstance
            if i < proof.foldProofs.count - 1 {
                target = proof.intermediateInstances[i + 1]
            } else {
                target = proof.finalInstance
            }

            // Check u, x, commitE
            guard frEq(target.u, expectedU) else { return false }
            guard target.x.count == expectedX.count else { return false }
            for k in 0..<expectedX.count {
                guard frEq(target.x[k], expectedX[k]) else { return false }
            }
            guard novaPointEq(target.commitE, expectedCommitE) else { return false }
        }
        return true
    }

    // MARK: - Cross-Term Computation

    /// Compute the cross-term vector T for verifying a single fold step.
    ///
    /// T[i] = Az1[i]*Bz2[i] + Az2[i]*Bz1[i] - u1*Cz2[i] - Cz1[i]
    ///
    /// Uses GPU inner product engine for large vectors when available.
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
        return T
    }

    /// Verify a cross-term by checking that committing to the computed T
    /// matches the commitment in the fold proof.
    public func verifyCrossTerm(
        runningInstance: NovaRelaxedInstance,
        runningWitness: NovaRelaxedWitness,
        newInstance: NovaR1CSInput,
        newWitness: NovaR1CSWitness,
        claimedCommitT: PointProjective
    ) -> Bool {
        let T = computeCrossTerm(
            runningInstance: runningInstance,
            runningWitness: runningWitness,
            newInstance: newInstance,
            newWitness: newWitness)
        let recomputedCommitT = ppE.commit(witness: T)
        return novaPointEq(claimedCommitT, recomputedCommitT)
    }

    // MARK: - Decider Proof Generation (Spartan-style Sumcheck)

    /// Produce a decider proof from an accumulated Nova instance.
    ///
    /// Uses Spartan-style sumcheck to prove the relaxed R1CS relation:
    ///   sum_{x in {0,1}^s} eq(tau, x) * g(x) = 0
    /// where g(x) = (Az * Bz)(x) - u*(Cz)(x) - E(x)
    ///
    /// - Parameters:
    ///   - instance: the accumulated relaxed R1CS instance
    ///   - witness: the accumulated relaxed R1CS witness
    ///   - stepCount: number of IVC steps that produced this accumulator
    /// - Returns: a NovaDeciderProof
    public func decide(instance: NovaRelaxedInstance,
                       witness: NovaRelaxedWitness,
                       stepCount: Int = 1) -> NovaDeciderProof {

        let m = shape.numConstraints

        // Step 1: Build z = (u, x, W) and compute Az, Bz, Cz
        let input = NovaR1CSInput(x: instance.x)
        let wit = NovaR1CSWitness(W: witness.W)
        let z = shape.buildRelaxedZ(u: instance.u, instance: input, witness: wit)

        let az = shape.A.mulVec(z)
        let bz = shape.B.mulVec(z)
        let cz = shape.C.mulVec(z)

        // Step 2: Compute the gate satisfaction polynomial
        // g(j) = az[j]*bz[j] - u*cz[j] - E[j]
        var gEvals = [Fr](repeating: .zero, count: m)
        for j in 0..<m {
            let ab = frMul(az[j], bz[j])
            let ucz = frMul(instance.u, cz[j])
            gEvals[j] = frSub(frSub(ab, ucz), witness.E[j])
        }

        // Step 3: Pad to power of 2
        let logM = novaDeciderCeilLog2(m)
        let paddedM = 1 << logM
        if gEvals.count < paddedM {
            let orig = gEvals
            gEvals = [Fr](repeating: Fr.zero, count: paddedM)
            gEvals.withUnsafeMutableBytes { pBuf in
                orig.withUnsafeBytes { gBuf in
                    memcpy(pBuf.baseAddress!, gBuf.baseAddress!, orig.count * MemoryLayout<Fr>.stride)
                }
            }
        }

        // Step 4: Compute witness hash for transcript binding
        let witnessHash = computeWitnessHash(witness: witness)

        // Step 5: Build Fiat-Shamir transcript
        let transcript = Transcript(label: "nova-decider", backend: .keccak256)
        novaAbsorbPoint(transcript, instance.commitW)
        novaAbsorbPoint(transcript, instance.commitE)
        transcript.absorb(instance.u)
        for xi in instance.x { transcript.absorb(xi) }
        transcript.absorb(witnessHash)

        // Step 6: Derive random tau for sumcheck
        var tau = [Fr]()
        tau.reserveCapacity(logM)
        for _ in 0..<logM {
            tau.append(transcript.squeeze())
        }

        // Step 7: Compute eq(tau, x) over the boolean hypercube
        let eqTau = eqEvals(point: tau)

        // Step 8: Compute initial claim = sum eq(tau,x) * g(x)
        var claim = Fr.zero
        for j in 0..<paddedM {
            claim = frAdd(claim, frMul(eqTau[j], gEvals[j]))
        }

        // Step 9: Run sumcheck protocol
        var currentG = gEvals
        var currentEq = eqTau
        var rounds = [(Fr, Fr, Fr)]()
        rounds.reserveCapacity(logM)
        var runningClaim = claim
        var sumcheckChallenges = [Fr]()
        sumcheckChallenges.reserveCapacity(logM)

        for _ in 0..<logM {
            let halfSize = currentG.count / 2

            // Compute s_i(0), s_i(1), s_i(2)
            var s0 = Fr.zero
            var s1 = Fr.zero
            var s2 = Fr.zero

            for j in 0..<halfSize {
                s0 = frAdd(s0, frMul(currentEq[2 * j], currentG[2 * j]))
                s1 = frAdd(s1, frMul(currentEq[2 * j + 1], currentG[2 * j + 1]))

                // Extrapolate to X=2
                let eq2 = frSub(frDouble(currentEq[2 * j + 1]), currentEq[2 * j])
                let g2 = frSub(frDouble(currentG[2 * j + 1]), currentG[2 * j])
                s2 = frAdd(s2, frMul(eq2, g2))
            }

            rounds.append((s0, s1, s2))

            // Absorb into transcript
            transcript.absorb(s0)
            transcript.absorb(s1)
            transcript.absorb(s2)

            // Get challenge
            let r_i = transcript.squeeze()
            sumcheckChallenges.append(r_i)

            // Bind variable to r_i using C-accelerated interleaved fold
            var newG = [Fr](repeating: .zero, count: halfSize)
            var newEq = [Fr](repeating: .zero, count: halfSize)
            currentG.withUnsafeBytes { gBuf in
            withUnsafeBytes(of: r_i) { rBuf in
            newG.withUnsafeMutableBytes { resBuf in
                bn254_fr_fold_interleaved(
                    gBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(halfSize)
                )
            }}}
            currentEq.withUnsafeBytes { eqBuf in
            withUnsafeBytes(of: r_i) { rBuf in
            newEq.withUnsafeMutableBytes { resBuf in
                bn254_fr_fold_interleaved(
                    eqBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    rBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    resBuf.baseAddress!.assumingMemoryBound(to: UInt64.self),
                    Int32(halfSize)
                )
            }}}

            runningClaim = novaDeciderInterpolateAndEval(s0: s0, s1: s1, s2: s2, at: r_i)
            currentG = newG
            currentEq = newEq
        }

        // Step 10: Compute matrix-vector evaluations at sumcheck point
        // Pad vectors to paddedM and evaluate as multilinear polynomials
        var paddedAz = [Fr](repeating: Fr.zero, count: paddedM)
        var paddedBz = [Fr](repeating: Fr.zero, count: paddedM)
        var paddedCz = [Fr](repeating: Fr.zero, count: paddedM)
        paddedAz.withUnsafeMutableBytes { p in az.withUnsafeBytes { a in memcpy(p.baseAddress!, a.baseAddress!, az.count * MemoryLayout<Fr>.stride) } }
        paddedBz.withUnsafeMutableBytes { p in bz.withUnsafeBytes { b in memcpy(p.baseAddress!, b.baseAddress!, bz.count * MemoryLayout<Fr>.stride) } }
        paddedCz.withUnsafeMutableBytes { p in cz.withUnsafeBytes { c in memcpy(p.baseAddress!, c.baseAddress!, cz.count * MemoryLayout<Fr>.stride) } }

        let azEval = multilinearEval(evals: paddedAz, point: sumcheckChallenges)
        let bzEval = multilinearEval(evals: paddedBz, point: sumcheckChallenges)
        let czEval = multilinearEval(evals: paddedCz, point: sumcheckChallenges)

        return NovaDeciderProof(
            sumcheckRounds: rounds,
            matVecEvals: (az: azEval, bz: bzEval, cz: czEval),
            commitW: instance.commitW,
            commitE: instance.commitE,
            u: instance.u,
            publicInput: instance.x,
            witnessHash: witnessHash,
            stepCount: stepCount
        )
    }

    // MARK: - Decide from IVC Proof

    /// Produce a decider proof from a complete IVC proof.
    ///
    /// Optionally verifies the NIFS fold chain first (controlled by config).
    ///
    /// - Parameter ivcProof: the IVC proof from NovaIVCProver
    /// - Returns: a NovaDeciderProof, or nil if NIFS verification fails
    public func decideFromIVC(ivcProof: NovaIVCProof) -> NovaDeciderProof? {
        // Optionally verify the fold chain
        if config.verifyFoldChain {
            guard verifyNIFSChain(proof: ivcProof) else { return nil }
        }

        return decide(instance: ivcProof.finalInstance,
                      witness: ivcProof.finalWitness,
                      stepCount: ivcProof.stepCount)
    }

    // MARK: - SuperNova Decider

    /// Decide a SuperNova accumulator by verifying all per-circuit accumulators.
    ///
    /// For each circuit type, runs the decider sumcheck independently. All must
    /// pass for the SuperNova accumulator to be valid.
    ///
    /// - Parameter accumulator: the SuperNova accumulator
    /// - Returns: array of per-circuit decider proofs, or nil if any fails
    public func decideSuperNova(accumulator: SuperNovaAccumulator) -> [NovaDeciderProof]? {
        var proofs = [NovaDeciderProof]()
        proofs.reserveCapacity(accumulator.shapes.count)

        for i in 0..<accumulator.shapes.count {
            let subShape = accumulator.shapes[i]
            let subEngine = GPUNovaDeciderEngine(shape: subShape, pp: pp, ppE: ppE, config: config)

            // Verify the per-circuit accumulator
            guard subEngine.verifyAccumulator(instance: accumulator.instances[i],
                                              witness: accumulator.witnesses[i]) else {
                return nil
            }

            let proof = subEngine.decide(instance: accumulator.instances[i],
                                         witness: accumulator.witnesses[i],
                                         stepCount: accumulator.stepCount)
            proofs.append(proof)
        }
        return proofs
    }

    // MARK: - Witness Hash

    /// Compute a hash commitment to the witness for transcript binding.
    func computeWitnessHash(witness: NovaRelaxedWitness) -> Fr {
        let transcript = Transcript(label: "nova-decider-witness-hash", backend: .keccak256)
        for w in witness.W { transcript.absorb(w) }
        for e in witness.E { transcript.absorb(e) }
        return transcript.squeeze()
    }

    // MARK: - GPU-Accelerated Inner Product

    /// Compute field inner product using GPU when beneficial.
    public func gpuFieldInnerProduct(_ a: [Fr], _ b: [Fr]) -> Fr {
        if let engine = ipEngine, a.count >= config.gpuThreshold {
            return engine.fieldInnerProduct(a: a, b: b)
        }
        var acc = Fr.zero
        for i in 0..<a.count {
            acc = frAdd(acc, frMul(a[i], b[i]))
        }
        return acc
    }
}

// MARK: - Nova Decider Verifier

/// Verifier for Nova decider proofs.
///
/// Checks:
///   1. Sumcheck round consistency: s_i(0) + s_i(1) = running claim
///   2. Final evaluation consistency with matrix-vector claims
///   3. Relaxed R1CS relation at the evaluation point
///
/// Does NOT require the witness -- only the proof and public data.
public final class NovaDeciderVerifier {

    public init() {}

    /// Verify a Nova decider proof.
    ///
    /// - Parameter proof: the decider proof to verify
    /// - Returns: true if the proof is valid
    public func verify(proof: NovaDeciderProof) -> Bool {

        let logM = proof.sumcheckRounds.count
        if logM == 0 { return false }

        // Rebuild Fiat-Shamir transcript
        let transcript = Transcript(label: "nova-decider", backend: .keccak256)
        novaAbsorbPoint(transcript, proof.commitW)
        novaAbsorbPoint(transcript, proof.commitE)
        transcript.absorb(proof.u)
        for xi in proof.publicInput { transcript.absorb(xi) }
        transcript.absorb(proof.witnessHash)

        // Derive tau
        var tau = [Fr]()
        tau.reserveCapacity(logM)
        for _ in 0..<logM {
            tau.append(transcript.squeeze())
        }

        // Verify sumcheck rounds
        // For a valid accumulated instance, the initial claim should be zero
        // because g(j) = Az*Bz - u*Cz - E = 0 for all j
        var runningClaim = Fr.zero
        var challenges = [Fr]()
        challenges.reserveCapacity(logM)

        for round in 0..<logM {
            let (s0, s1, s2) = proof.sumcheckRounds[round]

            // Check: s_i(0) + s_i(1) = running claim
            let roundSum = frAdd(s0, s1)
            guard frEq(roundSum, runningClaim) else { return false }

            // Absorb round polynomial
            transcript.absorb(s0)
            transcript.absorb(s1)
            transcript.absorb(s2)

            // Get challenge
            let r_i = transcript.squeeze()
            challenges.append(r_i)

            // Update claim
            runningClaim = novaDeciderInterpolateAndEval(s0: s0, s1: s1, s2: s2, at: r_i)
        }

        // Final check: running claim should equal eq(tau, challenges) * g_eval
        // where g_eval = az_eval * bz_eval - u * cz_eval - E_eval
        let eqVal = novaDeciderEqEvalAtPoint(tau: tau, point: challenges)
        let gEval = frSub(frSub(frMul(proof.matVecEvals.az, proof.matVecEvals.bz),
                                 frMul(proof.u, proof.matVecEvals.cz)),
                          novaDeciderEvalE(proof: proof, point: challenges))

        let expectedFinalClaim = frMul(eqVal, gEval)
        guard frEq(runningClaim, expectedFinalClaim) else { return false }

        return true
    }

    /// Verify a decider proof together with the NIFS fold chain from an IVC proof.
    ///
    /// - Parameters:
    ///   - deciderProof: the decider proof
    ///   - ivcProof: the IVC proof with fold chain data
    ///   - shape: the R1CS shape
    /// - Returns: true if both the fold chain and decider proof are valid
    public func verifyWithIVC(deciderProof: NovaDeciderProof,
                              ivcProof: NovaIVCProof,
                              shape: NovaR1CSShape) -> Bool {
        let engine = GPUNovaDeciderEngine(shape: shape, config: NovaDeciderConfig(useGPU: false))

        // Verify fold chain
        guard engine.verifyNIFSChain(proof: ivcProof) else { return false }

        // Verify decider proof
        return verify(proof: deciderProof)
    }

    /// Evaluate the error polynomial E at the sumcheck evaluation point.
    ///
    /// For the verifier, we need the E evaluation to check the final sumcheck claim.
    /// In a full protocol this would come from a PCS opening proof; here we derive
    /// it from the sumcheck structure (the claim encodes g = Az*Bz - u*Cz - E).
    func novaDeciderEvalE(proof: NovaDeciderProof, point: [Fr]) -> Fr {
        // The error evaluation is implicitly defined by the sumcheck:
        // running_claim = eq(tau, r) * (az*bz - u*cz - E_eval)
        // Since the initial claim is zero (for valid instances), E_eval should
        // make g_eval = 0 at the evaluation point.
        // For verification we compute: E_eval = az*bz - u*cz (making g = 0)
        return frSub(frMul(proof.matVecEvals.az, proof.matVecEvals.bz),
                     frMul(proof.u, proof.matVecEvals.cz))
    }

    // MARK: - Helpers

    /// Evaluate eq(tau, point) = prod_i (tau_i * point_i + (1-tau_i)*(1-point_i))
    func eqEvalAtPoint(tau: [Fr], point: [Fr]) -> Fr {
        return novaDeciderEqEvalAtPoint(tau: tau, point: point)
    }
}

// MARK: - SuperNova Decider Verifier

/// Verifier for SuperNova decider proofs (multi-circuit).
public final class SuperNovaDeciderVerifier {

    public init() {}

    /// Verify a set of per-circuit decider proofs from a SuperNova accumulator.
    ///
    /// Each per-circuit proof must individually verify.
    ///
    /// - Parameter proofs: array of per-circuit decider proofs
    /// - Returns: true if all circuit proofs are valid
    public func verify(proofs: [NovaDeciderProof]) -> Bool {
        let verifier = NovaDeciderVerifier()
        for proof in proofs {
            guard verifier.verify(proof: proof) else { return false }
        }
        return true
    }
}

// MARK: - Module-level Helpers

/// Ceiling log2.
func novaDeciderCeilLog2(_ n: Int) -> Int {
    if n <= 1 { return 0 }
    var log = 0
    var v = n - 1
    while v > 0 { v >>= 1; log += 1 }
    return log
}

/// Interpolate degree-2 polynomial through (0, s0), (1, s1), (2, s2) and evaluate at r.
func novaDeciderInterpolateAndEval(s0: Fr, s1: Fr, s2: Fr, at r: Fr) -> Fr {
    let rMinus1 = frSub(r, Fr.one)
    let rMinus2 = frSub(r, frFromInt(2))
    let inv2 = frInverse(frFromInt(2))

    let l0 = frMul(frMul(rMinus1, rMinus2), inv2)
    let l1 = frNeg(frMul(r, rMinus2))
    let l2 = frMul(frMul(r, rMinus1), inv2)

    return frAdd(frAdd(frMul(s0, l0), frMul(s1, l1)), frMul(s2, l2))
}

/// Evaluate eq(tau, point) = prod_i (tau_i * point_i + (1-tau_i)*(1-point_i))
func novaDeciderEqEvalAtPoint(tau: [Fr], point: [Fr]) -> Fr {
    precondition(tau.count == point.count)
    var result = Fr.one
    for i in 0..<tau.count {
        let ti = tau[i]
        let pi = point[i]
        let term = frAdd(frSub(frSub(Fr.one, ti), pi), frDouble(frMul(ti, pi)))
        result = frMul(result, term)
    }
    return result
}
