// GPU-Accelerated Nova Folding Verifier Engine
//
// Verifies Nova IVC folding proofs without requiring full witness access.
// Supports relaxed R1CS instance verification, cross-term checking,
// running instance accumulation verification, hash-based state consistency,
// batch verification of multiple folding steps, and SuperNova (multi-instruction)
// folding verification.
//
// Architecture:
//   GPUNovaVerifierEngine  -- top-level verifier engine
//     - Relaxed R1CS satisfaction check (with witness, for honest verifier)
//     - Cross-term recomputation and commitment verification
//     - NIFS fold chain replay with Fiat-Shamir transcript
//     - Hash-based state consistency checks (Poseidon2 binding)
//     - Batch verification of multi-step fold chains
//     - SuperNova multi-circuit accumulator verification
//
//   NovaVerifierTranscript -- deterministic Fiat-Shamir transcript for verifier
//   NovaFoldChainResult   -- result of a fold chain verification
//   NovaBatchVerifyResult -- result of batch verification across multiple proofs
//
// GPU acceleration targets:
//   - Inner product for large cross-term recomputation
//   - Sparse matrix-vector products for relaxed R1CS checks
//   - Batch Pedersen commitment verification
//
// Reference: "Nova: Recursive Zero-Knowledge Arguments from Folding Schemes"
//            (Kothapalli, Setty, Tzialla 2022)
// Reference: "SuperNova: Proving universal machine execution"
//            (Kothapalli, Setty 2022)

import Foundation
import Metal
import NeonFieldOps

// MARK: - Fold Chain Verification Result

/// Detailed result from verifying a Nova fold chain.
/// Tracks per-step outcomes so callers can identify which step failed.
public struct NovaFoldChainResult {
    /// Whether the entire fold chain is valid.
    public let valid: Bool
    /// Number of fold steps verified.
    public let stepsVerified: Int
    /// Index of the first failing step (-1 if all passed).
    public let failingStep: Int
    /// Hash of the final accumulated instance (for state binding).
    public let finalStateHash: Fr
    /// The re-derived challenges from the Fiat-Shamir transcript.
    public let challenges: [Fr]

    public init(valid: Bool, stepsVerified: Int, failingStep: Int,
                finalStateHash: Fr, challenges: [Fr]) {
        self.valid = valid
        self.stepsVerified = stepsVerified
        self.failingStep = failingStep
        self.finalStateHash = finalStateHash
        self.challenges = challenges
    }
}

// MARK: - Batch Verification Result

/// Result from batch-verifying multiple Nova IVC proofs.
public struct NovaBatchVerifyResult {
    /// Whether all proofs in the batch are valid.
    public let allValid: Bool
    /// Per-proof validity flags (true = valid).
    public let perProofValid: [Bool]
    /// Number of proofs that passed verification.
    public let passCount: Int
    /// Number of proofs that failed verification.
    public let failCount: Int

    public init(allValid: Bool, perProofValid: [Bool], passCount: Int, failCount: Int) {
        self.allValid = allValid
        self.perProofValid = perProofValid
        self.passCount = passCount
        self.failCount = failCount
    }
}

// MARK: - Verifier Configuration

/// Configuration for the Nova verifier engine.
public struct NovaVerifierConfig {
    /// Whether to use GPU acceleration for large field operations.
    public let useGPU: Bool
    /// Minimum vector size for GPU dispatch.
    public let gpuThreshold: Int
    /// Whether to verify Pedersen commitment openings (expensive).
    public let verifyCommitments: Bool
    /// Whether to compute and verify state hashes at each fold step.
    public let verifyStateHashes: Bool
    /// Whether to perform cross-term recomputation (requires witness).
    public let verifyCrossTerms: Bool

    public init(useGPU: Bool = true, gpuThreshold: Int = 512,
                verifyCommitments: Bool = true, verifyStateHashes: Bool = true,
                verifyCrossTerms: Bool = true) {
        self.useGPU = useGPU
        self.gpuThreshold = gpuThreshold
        self.verifyCommitments = verifyCommitments
        self.verifyStateHashes = verifyStateHashes
        self.verifyCrossTerms = verifyCrossTerms
    }
}

// MARK: - SuperNova Verification Result

/// Result from verifying a SuperNova multi-circuit accumulator.
public struct SuperNovaVerifyResult {
    /// Whether all per-circuit accumulators are valid.
    public let allValid: Bool
    /// Per-circuit validity flags.
    public let perCircuitValid: [Bool]
    /// Per-circuit state hashes for binding.
    public let perCircuitStateHashes: [Fr]
    /// Combined state hash across all circuits.
    public let combinedStateHash: Fr

    public init(allValid: Bool, perCircuitValid: [Bool],
                perCircuitStateHashes: [Fr], combinedStateHash: Fr) {
        self.allValid = allValid
        self.perCircuitValid = perCircuitValid
        self.perCircuitStateHashes = perCircuitStateHashes
        self.combinedStateHash = combinedStateHash
    }
}

// MARK: - GPU Nova Verifier Engine

/// GPU-accelerated Nova folding verifier engine.
///
/// Provides comprehensive verification of Nova IVC folding proofs:
///   - Relaxed R1CS satisfaction (honest verifier, with witness)
///   - NIFS fold chain replay (Fiat-Shamir transcript re-derivation)
///   - Cross-term recomputation and commitment check
///   - Hash-based state consistency (Poseidon2 binding)
///   - Batch verification of multiple proofs
///   - SuperNova multi-circuit accumulator verification
///
/// Usage:
///   1. Create engine with an R1CS shape
///   2. Call verifyRelaxedR1CS() to check a single accumulated instance
///   3. Call verifyFoldChain() to replay and verify an entire fold chain
///   4. Call batchVerify() to verify multiple IVC proofs at once
///   5. Call verifySuperNova() for multi-circuit verification
public final class GPUNovaVerifierEngine {

    public static let version = Versions.gpuNovaVerifier

    public let shape: NovaR1CSShape
    public let config: NovaVerifierConfig
    public let pp: PedersenParams
    public let ppE: PedersenParams

    /// GPU inner product engine for accelerated field operations.
    private let ipEngine: GPUInnerProductEngine?
    /// Whether GPU is available and enabled.
    public let gpuAvailable: Bool

    // MARK: - Initialization

    /// Initialize with an R1CS shape, generating fresh Pedersen parameters.
    public init(shape: NovaR1CSShape, config: NovaVerifierConfig = NovaVerifierConfig()) {
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
                config: NovaVerifierConfig = NovaVerifierConfig()) {
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

    // MARK: - Relaxed R1CS Verification

    /// Verify that a relaxed R1CS instance satisfies the relation: Az . Bz = u*(Cz) + E.
    ///
    /// This is the "honest verifier" check that requires the full witness.
    ///
    /// - Parameters:
    ///   - instance: the relaxed R1CS instance
    ///   - witness: the relaxed witness (W and E vectors)
    /// - Returns: true if the relaxed R1CS relation holds
    public func verifyRelaxedR1CS(instance: NovaRelaxedInstance,
                                   witness: NovaRelaxedWitness) -> Bool {
        return shape.satisfiesRelaxed(instance: instance, witness: witness)
    }

    /// Verify strict R1CS satisfaction: Az . Bz = Cz where z = (1, x, W).
    ///
    /// - Parameters:
    ///   - instance: the strict R1CS input
    ///   - witness: the strict R1CS witness
    /// - Returns: true if the strict R1CS relation holds
    public func verifyStrictR1CS(instance: NovaR1CSInput,
                                  witness: NovaR1CSWitness) -> Bool {
        return shape.satisfies(instance: instance, witness: witness)
    }

    // MARK: - Commitment Verification

    /// Verify that the commitments in a relaxed instance match the witness.
    ///
    /// Checks:
    ///   - commitW == Pedersen(W)
    ///   - commitE == Pedersen(E) (if E is non-zero)
    ///
    /// - Parameters:
    ///   - instance: the relaxed instance containing commitments
    ///   - witness: the full witness (W and E)
    /// - Returns: true if all commitments are consistent
    public func verifyCommitments(instance: NovaRelaxedInstance,
                                   witness: NovaRelaxedWitness) -> Bool {
        let recomputedW = pp.commit(witness: witness.W)
        guard novaPointEq(instance.commitW, recomputedW) else { return false }

        if witness.E.contains(where: { !$0.isZero }) {
            let recomputedE = ppE.commit(witness: witness.E)
            guard novaPointEq(instance.commitE, recomputedE) else { return false }
        }
        return true
    }

    // MARK: - Cross-Term Verification

    /// Recompute the cross-term vector T and verify its commitment.
    ///
    /// T[i] = Az1[i]*Bz2[i] + Az2[i]*Bz1[i] - u1*Cz2[i] - Cz1[i]
    ///
    /// - Parameters:
    ///   - runningInstance: the running relaxed instance
    ///   - runningWitness: the running relaxed witness
    ///   - newInstance: the fresh instance being folded in
    ///   - newWitness: the fresh witness
    /// - Returns: the recomputed cross-term vector T
    public func recomputeCrossTerm(
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

    /// Verify that a claimed cross-term commitment matches the recomputed cross-term.
    ///
    /// - Parameters:
    ///   - runningInstance: the running relaxed instance
    ///   - runningWitness: the running relaxed witness
    ///   - newInstance: the fresh instance
    ///   - newWitness: the fresh witness
    ///   - claimedCommitT: the prover's claimed commitment to T
    /// - Returns: true if the commitment matches
    public func verifyCrossTermCommitment(
        runningInstance: NovaRelaxedInstance,
        runningWitness: NovaRelaxedWitness,
        newInstance: NovaR1CSInput,
        newWitness: NovaR1CSWitness,
        claimedCommitT: PointProjective
    ) -> Bool {
        let T = recomputeCrossTerm(
            runningInstance: runningInstance,
            runningWitness: runningWitness,
            newInstance: newInstance,
            newWitness: newWitness)
        let recomputedCommitT = ppE.commit(witness: T)
        return novaPointEq(claimedCommitT, recomputedCommitT)
    }

    // MARK: - Fold Step Verification (without witness)

    /// Verify a single fold step using only public data (no witness required).
    ///
    /// Re-derives the Fiat-Shamir challenge r and checks that the target folded
    /// instance matches the expected linear combination:
    ///   - u' = u + r
    ///   - x' = x1 + r * x2
    ///   - commitE' = commitE + r * commitT
    ///
    /// - Parameters:
    ///   - running: the running relaxed instance before this fold
    ///   - fresh: the fresh instance being folded in
    ///   - proof: the fold proof (commitment to T)
    ///   - target: the claimed folded result
    /// - Returns: true if the fold step is consistent
    public func verifyFoldStep(
        running: NovaRelaxedInstance,
        fresh: NovaR1CSInput,
        proof: NovaFoldProof,
        target: NovaRelaxedInstance
    ) -> Bool {
        // Re-derive Fiat-Shamir challenge
        let r = deriveVerifierChallenge(running: running, fresh: fresh, commitT: proof.commitT)

        // Check u' = u + r
        let expectedU = frAdd(running.u, r)
        guard frEq(target.u, expectedU) else { return false }

        // Check x' = x1 + r * x2
        guard target.x.count == running.x.count else { return false }
        for k in 0..<running.x.count {
            let expected = frAdd(running.x[k], frMul(r, fresh.x[k]))
            guard frEq(target.x[k], expected) else { return false }
        }

        // Check commitE' = commitE + r * commitT
        let expectedCommitE = pointAdd(running.commitE,
                                        cPointScalarMul(proof.commitT, r))
        guard novaPointEq(target.commitE, expectedCommitE) else { return false }

        return true
    }

    // MARK: - Fold Chain Verification

    /// Verify an entire NIFS fold chain from an IVC proof.
    ///
    /// Replays the Fiat-Shamir transcript for each fold step and checks
    /// consistency of (u, x, commitE) across all steps.
    ///
    /// Returns a detailed result including per-step outcomes.
    ///
    /// - Parameter proof: the IVC proof containing fold chain data
    /// - Returns: NovaFoldChainResult with verification outcome
    public func verifyFoldChain(proof: NovaIVCProof) -> NovaFoldChainResult {
        // Single-step proof: trivially valid
        if proof.stepCount <= 1 {
            let stateHash = hashRelaxedInstance(proof.finalInstance)
            return NovaFoldChainResult(
                valid: true, stepsVerified: 1, failingStep: -1,
                finalStateHash: stateHash, challenges: [])
        }

        guard proof.foldProofs.count == proof.stepCount - 1,
              proof.freshInstances.count == proof.stepCount - 1,
              proof.intermediateInstances.count == proof.stepCount - 1 else {
            return NovaFoldChainResult(
                valid: false, stepsVerified: 0, failingStep: 0,
                finalStateHash: Fr.zero, challenges: [])
        }

        var challenges = [Fr]()
        challenges.reserveCapacity(proof.foldProofs.count)

        for i in 0..<proof.foldProofs.count {
            let running = proof.intermediateInstances[i]
            let fresh = proof.freshInstances[i]
            let foldProof = proof.foldProofs[i]

            // Determine target
            let target: NovaRelaxedInstance
            if i < proof.foldProofs.count - 1 {
                target = proof.intermediateInstances[i + 1]
            } else {
                target = proof.finalInstance
            }

            // Re-derive challenge
            let r = deriveVerifierChallenge(running: running, fresh: fresh,
                                            commitT: foldProof.commitT)
            challenges.append(r)

            // Check u' = u + r
            let expectedU = frAdd(running.u, r)
            guard frEq(target.u, expectedU) else {
                return NovaFoldChainResult(
                    valid: false, stepsVerified: i, failingStep: i,
                    finalStateHash: Fr.zero, challenges: challenges)
            }

            // Check x' = x1 + r * x2
            guard target.x.count == running.x.count else {
                return NovaFoldChainResult(
                    valid: false, stepsVerified: i, failingStep: i,
                    finalStateHash: Fr.zero, challenges: challenges)
            }
            for k in 0..<running.x.count {
                let expected = frAdd(running.x[k], frMul(r, fresh.x[k]))
                guard frEq(target.x[k], expected) else {
                    return NovaFoldChainResult(
                        valid: false, stepsVerified: i, failingStep: i,
                        finalStateHash: Fr.zero, challenges: challenges)
                }
            }

            // Check commitE' = commitE + r * commitT
            let expectedCommitE = pointAdd(running.commitE,
                                            cPointScalarMul(foldProof.commitT, r))
            guard novaPointEq(target.commitE, expectedCommitE) else {
                return NovaFoldChainResult(
                    valid: false, stepsVerified: i, failingStep: i,
                    finalStateHash: Fr.zero, challenges: challenges)
            }
        }

        let finalHash = hashRelaxedInstance(proof.finalInstance)
        return NovaFoldChainResult(
            valid: true, stepsVerified: proof.foldProofs.count, failingStep: -1,
            finalStateHash: finalHash, challenges: challenges)
    }

    // MARK: - Running Instance Accumulation Verification

    /// Verify that folding two instances produces the expected accumulated result.
    ///
    /// Given the inputs and the claimed folded output, recomputes the fold
    /// and checks equality. Requires witness for cross-term computation.
    ///
    /// - Parameters:
    ///   - runningInstance: the running relaxed instance
    ///   - runningWitness: the running relaxed witness
    ///   - newInstance: the fresh instance
    ///   - newWitness: the fresh witness
    ///   - claimedResult: the claimed folded instance
    ///   - claimedWitness: the claimed folded witness
    /// - Returns: true if the claimed result matches the recomputed fold
    public func verifyAccumulation(
        runningInstance: NovaRelaxedInstance,
        runningWitness: NovaRelaxedWitness,
        newInstance: NovaR1CSInput,
        newWitness: NovaR1CSWitness,
        claimedResult: NovaRelaxedInstance,
        claimedWitness: NovaRelaxedWitness
    ) -> Bool {
        // Recompute the cross-term
        let T = recomputeCrossTerm(
            runningInstance: runningInstance,
            runningWitness: runningWitness,
            newInstance: newInstance,
            newWitness: newWitness)

        // Commit to T
        let commitT = ppE.commit(witness: T)

        // Derive challenge
        let r = deriveVerifierChallenge(
            running: runningInstance, fresh: newInstance, commitT: commitT)

        // Check u
        let expectedU = frAdd(runningInstance.u, r)
        guard frEq(claimedResult.u, expectedU) else { return false }

        // Check x
        guard claimedResult.x.count == runningInstance.x.count else { return false }
        for k in 0..<runningInstance.x.count {
            let expected = frAdd(runningInstance.x[k], frMul(r, newInstance.x[k]))
            guard frEq(claimedResult.x[k], expected) else { return false }
        }

        // Check witness W' = W1 + r * W2
        guard claimedWitness.W.count == runningWitness.W.count else { return false }
        for k in 0..<runningWitness.W.count {
            let expected = frAdd(runningWitness.W[k], frMul(r, newWitness.W[k]))
            guard frEq(claimedWitness.W[k], expected) else { return false }
        }

        // Check error E' = E1 + r * T
        guard claimedWitness.E.count == runningWitness.E.count else { return false }
        for k in 0..<runningWitness.E.count {
            let expected = frAdd(runningWitness.E[k], frMul(r, T[k]))
            guard frEq(claimedWitness.E[k], expected) else { return false }
        }

        // Verify the folded result satisfies relaxed R1CS
        return shape.satisfiesRelaxed(instance: claimedResult, witness: claimedWitness)
    }

    // MARK: - Hash-Based State Verification

    /// Compute a Poseidon2 hash binding a relaxed R1CS instance.
    ///
    /// Hashes: u || x[0] || x[1] || ... || x[l-1]
    /// This binds the public state of the accumulated instance.
    ///
    /// - Parameter instance: the relaxed instance to hash
    /// - Returns: the Poseidon2 hash of the instance's public data
    public func hashRelaxedInstance(_ instance: NovaRelaxedInstance) -> Fr {
        var inputs = [Fr]()
        inputs.reserveCapacity(1 + instance.x.count)
        inputs.append(instance.u)
        inputs.append(contentsOf: instance.x)
        return poseidon2HashMany(inputs)
    }

    /// Verify that a claimed state hash matches a relaxed instance.
    ///
    /// - Parameters:
    ///   - instance: the relaxed instance
    ///   - claimedHash: the claimed Poseidon2 hash
    /// - Returns: true if the hash matches
    public func verifyStateHash(instance: NovaRelaxedInstance,
                                 claimedHash: Fr) -> Bool {
        let computed = hashRelaxedInstance(instance)
        return frEq(computed, claimedHash)
    }

    /// Compute a chained state hash over a sequence of relaxed instances.
    ///
    /// H_0 = hash(instance_0)
    /// H_i = poseidon2(H_{i-1}, hash(instance_i))
    ///
    /// This creates a Merkle-like chain binding all intermediate states.
    ///
    /// - Parameter instances: the sequence of relaxed instances
    /// - Returns: the chained hash
    public func chainedStateHash(_ instances: [NovaRelaxedInstance]) -> Fr {
        guard !instances.isEmpty else { return Fr.zero }
        var acc = hashRelaxedInstance(instances[0])
        for i in 1..<instances.count {
            let h = hashRelaxedInstance(instances[i])
            acc = poseidon2Hash(acc, h)
        }
        return acc
    }

    // MARK: - Batch Verification

    /// Batch-verify multiple IVC proofs.
    ///
    /// Verifies each proof's fold chain independently and returns a summary.
    /// All proofs must use the same R1CS shape as this engine.
    ///
    /// - Parameter proofs: array of IVC proofs to verify
    /// - Returns: NovaBatchVerifyResult with per-proof outcomes
    public func batchVerify(proofs: [NovaIVCProof]) -> NovaBatchVerifyResult {
        var perProofValid = [Bool]()
        perProofValid.reserveCapacity(proofs.count)
        var passCount = 0
        var failCount = 0

        for proof in proofs {
            let chainResult = verifyFoldChain(proof: proof)
            if chainResult.valid {
                // Also check final relaxed R1CS satisfaction
                let relaxedOk = shape.satisfiesRelaxed(
                    instance: proof.finalInstance,
                    witness: proof.finalWitness)
                if relaxedOk {
                    perProofValid.append(true)
                    passCount += 1
                } else {
                    perProofValid.append(false)
                    failCount += 1
                }
            } else {
                perProofValid.append(false)
                failCount += 1
            }
        }

        return NovaBatchVerifyResult(
            allValid: failCount == 0,
            perProofValid: perProofValid,
            passCount: passCount,
            failCount: failCount)
    }

    /// Batch-verify multiple individual fold steps (no full chain required).
    ///
    /// Checks that each (running, fresh, proof, target) tuple is consistent.
    ///
    /// - Parameter steps: array of fold step data
    /// - Returns: per-step validity flags
    public func batchVerifyFoldSteps(
        steps: [(running: NovaRelaxedInstance, fresh: NovaR1CSInput,
                 proof: NovaFoldProof, target: NovaRelaxedInstance)]
    ) -> [Bool] {
        return steps.map { step in
            verifyFoldStep(running: step.running, fresh: step.fresh,
                          proof: step.proof, target: step.target)
        }
    }

    // MARK: - SuperNova Verification

    /// Verify a SuperNova multi-circuit accumulator.
    ///
    /// For each circuit type, checks that the accumulated instance satisfies
    /// the relaxed R1CS relation with respect to its own shape. Also computes
    /// per-circuit and combined state hashes for binding.
    ///
    /// - Parameter accumulator: the SuperNova accumulator
    /// - Returns: SuperNovaVerifyResult with per-circuit outcomes
    public func verifySuperNova(accumulator: SuperNovaAccumulator) -> SuperNovaVerifyResult {
        let numCircuits = accumulator.shapes.count
        var perCircuitValid = [Bool]()
        perCircuitValid.reserveCapacity(numCircuits)
        var perCircuitHashes = [Fr]()
        perCircuitHashes.reserveCapacity(numCircuits)
        var allValid = true

        for i in 0..<numCircuits {
            let subShape = accumulator.shapes[i]
            let inst = accumulator.instances[i]
            let wit = accumulator.witnesses[i]

            // Check relaxed R1CS satisfaction for this circuit's accumulator
            let ok = subShape.satisfiesRelaxed(instance: inst, witness: wit)
            perCircuitValid.append(ok)
            if !ok { allValid = false }

            // Compute state hash for this circuit
            let h = hashRelaxedInstance(inst)
            perCircuitHashes.append(h)
        }

        // Compute combined state hash
        var combined = Fr.zero
        for h in perCircuitHashes {
            combined = poseidon2Hash(combined, h)
        }
        // Also fold in the circuit schedule
        for idx in accumulator.circuitSchedule {
            combined = poseidon2Hash(combined, frFromInt(UInt64(idx)))
        }

        return SuperNovaVerifyResult(
            allValid: allValid,
            perCircuitValid: perCircuitValid,
            perCircuitStateHashes: perCircuitHashes,
            combinedStateHash: combined)
    }

    /// Verify a SuperNova accumulator with commitment checks.
    ///
    /// In addition to relaxed R1CS checks, verifies that the Pedersen
    /// commitments in each circuit's instance match its witness.
    ///
    /// - Parameters:
    ///   - accumulator: the SuperNova accumulator
    ///   - ppPerCircuit: Pedersen params for each circuit (witness commitment)
    ///   - ppEPerCircuit: Pedersen params for each circuit (error commitment)
    /// - Returns: true if all circuits pass all checks
    public func verifySuperNovaWithCommitments(
        accumulator: SuperNovaAccumulator,
        ppPerCircuit: [PedersenParams],
        ppEPerCircuit: [PedersenParams]
    ) -> Bool {
        guard ppPerCircuit.count == accumulator.shapes.count,
              ppEPerCircuit.count == accumulator.shapes.count else {
            return false
        }

        for i in 0..<accumulator.shapes.count {
            let subShape = accumulator.shapes[i]
            let inst = accumulator.instances[i]
            let wit = accumulator.witnesses[i]

            // Relaxed R1CS check
            guard subShape.satisfiesRelaxed(instance: inst, witness: wit) else { return false }

            // Witness commitment check
            let recomputedW = ppPerCircuit[i].commit(witness: wit.W)
            guard novaPointEq(inst.commitW, recomputedW) else { return false }

            // Error commitment check (if E is non-zero)
            if wit.E.contains(where: { !$0.isZero }) {
                let recomputedE = ppEPerCircuit[i].commit(witness: wit.E)
                guard novaPointEq(inst.commitE, recomputedE) else { return false }
            }
        }
        return true
    }

    // MARK: - Full IVC Verification

    /// Perform a comprehensive IVC proof verification.
    ///
    /// Combines:
    ///   1. Fold chain replay (Fiat-Shamir consistency)
    ///   2. Final relaxed R1CS satisfaction check
    ///   3. Commitment verification (optional, controlled by config)
    ///   4. State hash computation and verification
    ///
    /// - Parameters:
    ///   - proof: the IVC proof
    ///   - expectedStateHash: optional expected final state hash
    /// - Returns: true if all checks pass
    public func verifyIVC(proof: NovaIVCProof,
                           expectedStateHash: Fr? = nil) -> Bool {
        // Step 1: Verify fold chain
        let chainResult = verifyFoldChain(proof: proof)
        guard chainResult.valid else { return false }

        // Step 2: Verify final relaxed R1CS
        guard shape.satisfiesRelaxed(instance: proof.finalInstance,
                                      witness: proof.finalWitness) else {
            return false
        }

        // Step 3: Commitment verification (if enabled)
        if config.verifyCommitments {
            guard verifyCommitments(instance: proof.finalInstance,
                                    witness: proof.finalWitness) else {
                return false
            }
        }

        // Step 4: State hash verification (if expected hash provided)
        if let expected = expectedStateHash {
            guard verifyStateHash(instance: proof.finalInstance,
                                   claimedHash: expected) else {
                return false
            }
        }

        return true
    }

    // MARK: - Derive Verifier Challenge (Fiat-Shamir)

    /// Re-derive the folding challenge r from the verifier's perspective.
    ///
    /// Uses the same Fiat-Shamir transcript as the prover to ensure consistency.
    ///
    /// - Parameters:
    ///   - running: the running relaxed instance
    ///   - fresh: the fresh instance
    ///   - commitT: commitment to the cross-term
    /// - Returns: the derived challenge r
    public func deriveVerifierChallenge(
        running: NovaRelaxedInstance,
        fresh: NovaR1CSInput,
        commitT: PointProjective
    ) -> Fr {
        let transcript = Transcript(label: "nova-r1cs-fold", backend: .keccak256)

        novaAbsorbPoint(transcript, running.commitW)
        novaAbsorbPoint(transcript, running.commitE)
        transcript.absorb(running.u)
        for xi in running.x { transcript.absorb(xi) }

        for xi in fresh.x { transcript.absorb(xi) }

        novaAbsorbPoint(transcript, commitT)

        return transcript.squeeze()
    }

    // MARK: - GPU-Accelerated Inner Product

    /// Compute field inner product using GPU when beneficial.
    ///
    /// Falls back to CPU for small vectors or when GPU is unavailable.
    ///
    /// - Parameters:
    ///   - a: first vector
    ///   - b: second vector
    /// - Returns: sum of a[i] * b[i]
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

    // MARK: - Error Vector Analysis

    /// Compute the L2 "norm" of the error vector (sum of E[i]^2).
    ///
    /// Useful for monitoring accumulator health: a well-formed accumulator
    /// should have controlled error growth.
    ///
    /// - Parameter witness: the relaxed witness containing E
    /// - Returns: sum of E[i]^2 in the field
    public func errorNormSquared(witness: NovaRelaxedWitness) -> Fr {
        var acc = Fr.zero
        for e in witness.E {
            acc = frAdd(acc, frMul(e, e))
        }
        return acc
    }

    /// Check if the error vector is entirely zero.
    ///
    /// A zero error vector indicates this is a base (un-folded) instance.
    ///
    /// - Parameter witness: the relaxed witness
    /// - Returns: true if all E[i] == 0
    public func isBaseCase(witness: NovaRelaxedWitness) -> Bool {
        return witness.E.allSatisfy { $0.isZero }
    }
}
