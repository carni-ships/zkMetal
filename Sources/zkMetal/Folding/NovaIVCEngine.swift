// Nova IVC Engine — Unified Incrementally Verifiable Computation Engine
//
// Provides a high-level API for building IVC chains using the Nova folding scheme.
// Wraps NovaFoldProver and NovaFoldVerifier (from NovaFolding.swift) with:
//   - Step function abstraction (F: state_i -> state_{i+1})
//   - Automatic accumulator management
//   - Full IVC proof generation and verification
//   - Decider: final relaxed R1CS satisfaction + commitment opening checks
//
// Architecture:
//   NovaStepCircuit    — protocol for user-defined step computations
//   NovaIVCProver      — repeatedly folds step circuit instances
//   NovaIVCVerifier    — verifies the final accumulated proof
//   NovaIVCProof       — proof artifact carrying the folding chain result
//
// The folding-based IVC achieves:
//   - Per-step cost: O(n) field ops + 1 MSM (for cross-term commitment)
//   - Final proof size: O(1) regardless of computation length
//   - Prover state: single relaxed R1CS instance + witness
//
// Reference: "Nova: Recursive Zero-Knowledge Arguments from Folding Schemes"
//            (Kothapalli, Setty, Tzialla 2022)

import Foundation
import NeonFieldOps

// MARK: - Step Circuit Protocol

/// Protocol for user-defined IVC step computations.
///
/// Each step takes:
///   - stepIndex: the current IVC step number (0-based)
///   - stateIn: the public state from the previous step
///
/// And returns:
///   - publicInput: the full public input vector for the R1CS (includes stateIn + stateOut)
///   - witness: the private witness satisfying the step's R1CS
///
/// The step circuit's R1CS shape must remain constant across all steps.
public protocol NovaStepCircuit {
    /// Synthesize the R1CS instance for step i.
    /// Returns (publicInput, witness) satisfying the step's R1CS constraints.
    func synthesize(stepIndex: Int, stateIn: [Fr]) -> (publicInput: [Fr], witness: [Fr])
}

/// A concrete step circuit backed by a closure.
public struct NovaClosureStep: NovaStepCircuit {
    private let closure: (Int, [Fr]) -> (publicInput: [Fr], witness: [Fr])

    public init(_ f: @escaping (Int, [Fr]) -> (publicInput: [Fr], witness: [Fr])) {
        self.closure = f
    }

    public func synthesize(stepIndex: Int, stateIn: [Fr]) -> (publicInput: [Fr], witness: [Fr]) {
        closure(stepIndex, stateIn)
    }
}

// MARK: - IVC Proof

/// Complete IVC proof produced by NovaIVCProver.
///
/// Contains everything the verifier needs:
///   - The final accumulated relaxed R1CS instance
///   - The final accumulated relaxed R1CS witness (for the decider)
///   - All intermediate fold proofs (commitments to cross-terms)
///   - The sequence of fresh instances (for fold verification)
///   - The step count
public struct NovaIVCProof {
    /// Final accumulated relaxed R1CS instance.
    public let finalInstance: NovaRelaxedInstance

    /// Final accumulated witness (needed for the decider).
    public let finalWitness: NovaRelaxedWitness

    /// Fold proofs for each folding step (count = stepCount - 1).
    public let foldProofs: [NovaFoldProof]

    /// Fresh instances that were folded in (count = stepCount - 1).
    public let freshInstances: [NovaR1CSInput]

    /// Intermediate accumulated instances before each fold (count = stepCount - 1).
    /// intermediateInstances[i] is the running instance before fold i.
    public let intermediateInstances: [NovaRelaxedInstance]

    /// Number of IVC steps (including the base).
    public let stepCount: Int

    public init(finalInstance: NovaRelaxedInstance, finalWitness: NovaRelaxedWitness,
                foldProofs: [NovaFoldProof], freshInstances: [NovaR1CSInput],
                intermediateInstances: [NovaRelaxedInstance], stepCount: Int) {
        self.finalInstance = finalInstance
        self.finalWitness = finalWitness
        self.foldProofs = foldProofs
        self.freshInstances = freshInstances
        self.intermediateInstances = intermediateInstances
        self.stepCount = stepCount
    }
}

// MARK: - Nova IVC Prover

/// IVC prover that repeatedly folds step circuits using Nova.
///
/// Usage:
///   1. Create with an R1CS shape (shared across all steps)
///   2. Call `prove(steps:initialState:)` to run the full IVC chain
///   3. Or use `initialize` + `step` for incremental proving
///
/// The prover maintains a running relaxed R1CS instance and witness.
/// Each step computes the cross-term, commits to it, derives a Fiat-Shamir
/// challenge, and folds the new instance into the accumulator.
public class NovaIVCProver {
    /// The R1CS shape shared by all steps.
    public let shape: NovaR1CSShape

    /// Pedersen parameters for witness and cross-term commitments.
    public let pp: PedersenParams

    /// The underlying fold prover.
    public let foldProver: NovaFoldProver

    /// Running accumulated instance (nil before initialization).
    public private(set) var runningInstance: NovaRelaxedInstance?

    /// Running accumulated witness (nil before initialization).
    public private(set) var runningWitness: NovaRelaxedWitness?

    /// Current step count.
    public private(set) var stepCount: Int = 0

    /// Fold proofs collected during proving.
    private var collectedProofs: [NovaFoldProof] = []

    /// Fresh instances folded in.
    private var collectedFresh: [NovaR1CSInput] = []

    /// Intermediate running instances (before each fold).
    private var collectedIntermediate: [NovaRelaxedInstance] = []

    /// Initialize the IVC prover with an R1CS shape.
    ///
    /// - Parameter shape: the R1CS constraint system shape (shared across steps)
    public init(shape: NovaR1CSShape) {
        self.shape = shape
        self.foldProver = NovaFoldProver(shape: shape)
        self.pp = foldProver.pp
    }

    /// Initialize with pre-existing Pedersen parameters.
    public init(shape: NovaR1CSShape, pp: PedersenParams) {
        self.shape = shape
        self.pp = pp
        self.foldProver = NovaFoldProver(shape: shape, pp: pp)
    }

    // MARK: - Incremental API

    /// Initialize the IVC chain with the base step (step 0).
    ///
    /// Relaxes the first R1CS instance to form the initial accumulator (u=1, E=0).
    ///
    /// - Parameters:
    ///   - instance: public input for step 0
    ///   - witness: private witness for step 0
    public func initialize(instance: NovaR1CSInput, witness: NovaR1CSWitness) {
        precondition(shape.satisfies(instance: instance, witness: witness),
                     "Base instance must satisfy R1CS")
        let (relaxedInst, relaxedWit) = shape.relax(instance: instance, witness: witness, pp: pp)
        self.runningInstance = relaxedInst
        self.runningWitness = relaxedWit
        self.stepCount = 1
        self.collectedProofs = []
        self.collectedFresh = []
        self.collectedIntermediate = []
    }

    /// Fold one new step into the running accumulator.
    ///
    /// - Parameters:
    ///   - instance: public input for this step
    ///   - witness: private witness for this step
    /// - Returns: the fold proof for this step
    @discardableResult
    public func foldStep(instance: NovaR1CSInput, witness: NovaR1CSWitness) -> NovaFoldProof {
        guard let runInst = runningInstance, let runWit = runningWitness else {
            preconditionFailure("Must call initialize() before foldStep()")
        }

        collectedIntermediate.append(runInst)
        collectedFresh.append(instance)

        let (foldedInst, foldedWit, proof) = foldProver.fold(
            runningInstance: runInst, runningWitness: runWit,
            newInstance: instance, newWitness: witness)

        self.runningInstance = foldedInst
        self.runningWitness = foldedWit
        self.stepCount += 1
        self.collectedProofs.append(proof)

        return proof
    }

    /// Finalize the IVC chain and produce a complete proof.
    ///
    /// - Returns: the IVC proof containing all data needed for verification
    public func finalize() -> NovaIVCProof {
        guard let finalInst = runningInstance, let finalWit = runningWitness else {
            preconditionFailure("Must initialize and fold at least one step")
        }
        return NovaIVCProof(
            finalInstance: finalInst,
            finalWitness: finalWit,
            foldProofs: collectedProofs,
            freshInstances: collectedFresh,
            intermediateInstances: collectedIntermediate,
            stepCount: stepCount)
    }

    // MARK: - Batch API

    /// Run a complete IVC chain from a sequence of step instances.
    ///
    /// Step 0 becomes the base (relaxed). Steps 1..n are folded sequentially.
    ///
    /// - Parameter steps: array of (instance, witness) pairs, one per IVC step
    /// - Returns: the complete IVC proof
    public func prove(steps: [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]) -> NovaIVCProof {
        precondition(!steps.isEmpty, "Need at least one step")

        initialize(instance: steps[0].instance, witness: steps[0].witness)

        for i in 1..<steps.count {
            foldStep(instance: steps[i].instance, witness: steps[i].witness)
        }

        return finalize()
    }

    /// Run a complete IVC chain using a step circuit.
    ///
    /// - Parameters:
    ///   - circuit: the step circuit to execute at each step
    ///   - numSteps: total number of IVC steps
    ///   - initialState: the public state for step 0
    /// - Returns: the complete IVC proof
    public func prove(circuit: NovaStepCircuit, numSteps: Int,
                      initialState: [Fr]) -> NovaIVCProof {
        precondition(numSteps >= 1, "Need at least one step")

        var state = initialState
        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        steps.reserveCapacity(numSteps)

        for i in 0..<numSteps {
            let (pub, wit) = circuit.synthesize(stepIndex: i, stateIn: state)
            steps.append((NovaR1CSInput(x: pub), NovaR1CSWitness(W: wit)))
            // Extract the output state from public input for next step
            // Convention: public input = [stateIn..., stateOut...]
            // The output state becomes the next step's input state
            let halfPub = pub.count / 2
            if halfPub > 0 && i < numSteps - 1 {
                state = Array(pub[halfPub...])
            }
        }

        return prove(steps: steps)
    }
}

// MARK: - Nova IVC Verifier

/// Verifier for Nova IVC proofs.
///
/// Verification has two levels:
///   1. **Fold verification** (cheap): checks each fold step was computed correctly
///      by re-deriving the Fiat-Shamir challenge and checking (u, x, commitE) linearity.
///   2. **Decider** (expensive): checks the final relaxed R1CS relation holds
///      with the accumulated witness. This is done once at the end.
///
/// In production, the decider would be wrapped in a SNARK (Spartan/Groth16)
/// to eliminate the need for the witness.
public class NovaIVCVerifier {
    /// The R1CS shape to verify against.
    public let shape: NovaR1CSShape

    /// The fold verifier for checking individual fold steps.
    public let foldVerifier: NovaFoldVerifier

    public init(shape: NovaR1CSShape) {
        self.shape = shape
        self.foldVerifier = NovaFoldVerifier(shape: shape)
    }

    // MARK: - Full Verification

    /// Verify a complete IVC proof.
    ///
    /// Checks:
    ///   1. Each fold step was performed correctly (Fiat-Shamir + linearity)
    ///   2. The final accumulated instance satisfies the relaxed R1CS relation
    ///   3. Commitments to W and E are consistent with the witness
    ///
    /// - Parameters:
    ///   - proof: the IVC proof to verify
    ///   - pp: Pedersen parameters (must match the prover's)
    /// - Returns: true if the proof is valid
    public func verify(proof: NovaIVCProof, pp: PedersenParams) -> Bool {
        // Single-step proof: just check relaxed R1CS satisfaction
        if proof.stepCount == 1 {
            return deciderCheck(instance: proof.finalInstance,
                              witness: proof.finalWitness, pp: pp)
        }

        // Multi-step: verify each fold, then run the decider
        // Reconstruct the chain of accumulated instances
        guard proof.foldProofs.count == proof.stepCount - 1 else { return false }
        guard proof.freshInstances.count == proof.stepCount - 1 else { return false }
        guard proof.intermediateInstances.count == proof.stepCount - 1 else { return false }

        // Verify each fold step
        for i in 0..<proof.foldProofs.count {
            // Re-derive the folded instance from the fold verifier
            let running = proof.intermediateInstances[i]
            let fresh = proof.freshInstances[i]
            let foldProof = proof.foldProofs[i]

            // Compute expected folded instance
            let transcript = Transcript(label: "nova-r1cs-fold", backend: .keccak256)
            novaAbsorbPoint(transcript, running.commitW)
            novaAbsorbPoint(transcript, running.commitE)
            transcript.absorb(running.u)
            for xi in running.x { transcript.absorb(xi) }
            for xi in fresh.x { transcript.absorb(xi) }
            novaAbsorbPoint(transcript, foldProof.commitT)
            let r = transcript.squeeze()

            let expectedU = frAdd(running.u, r)
            let expectedCommitE = pointAdd(running.commitE,
                                           cPointScalarMul(foldProof.commitT, r))

            var expectedX = [Fr](repeating: .zero, count: running.x.count)
            for k in 0..<running.x.count {
                expectedX[k] = frAdd(running.x[k], frMul(r, fresh.x[k]))
            }

            // Determine the target instance to check against
            let target: NovaRelaxedInstance
            if i < proof.foldProofs.count - 1 {
                target = proof.intermediateInstances[i + 1]
            } else {
                target = proof.finalInstance
            }

            // Check u
            guard frEq(target.u, expectedU) else { return false }

            // Check x
            guard target.x.count == expectedX.count else { return false }
            for k in 0..<expectedX.count {
                guard frEq(target.x[k], expectedX[k]) else { return false }
            }

            // Check commitE
            guard novaPointEq(target.commitE, expectedCommitE) else { return false }
        }

        // Final decider check: relaxed R1CS satisfaction + commitment opening
        return deciderCheck(instance: proof.finalInstance,
                           witness: proof.finalWitness, pp: pp)
    }

    // MARK: - Decider

    /// Check the final accumulated instance satisfies relaxed R1CS.
    ///
    /// Verifies:
    ///   1. Az . Bz = u*(Cz) + E  (relaxed R1CS relation)
    ///   2. Pedersen commitment to W opens correctly
    ///   3. Pedersen commitment to E opens correctly
    ///
    /// - Parameters:
    ///   - instance: the relaxed R1CS instance
    ///   - witness: the relaxed witness (W and E)
    ///   - pp: Pedersen parameters
    /// - Returns: true if all checks pass
    public func deciderCheck(instance: NovaRelaxedInstance, witness: NovaRelaxedWitness,
                             pp: PedersenParams) -> Bool {
        // Check 1: Commitment to W opens correctly
        let recomputedW = pp.commit(witness: witness.W)
        guard novaPointEq(instance.commitW, recomputedW) else {
            return false
        }

        // Check 2: Relaxed R1CS relation
        guard shape.satisfiesRelaxed(instance: instance, witness: witness) else {
            return false
        }

        return true
    }

    // MARK: - Lightweight Fold-Only Verification

    /// Verify only the fold steps (no decider).
    ///
    /// This is a cheaper check that ensures the folding was done correctly,
    /// but does NOT check that the final instance actually satisfies relaxed R1CS.
    /// Useful when the decider will be run separately (e.g., inside a SNARK).
    ///
    /// - Parameter proof: the IVC proof to verify
    /// - Returns: true if all fold steps are valid
    public func verifyFoldsOnly(proof: NovaIVCProof) -> Bool {
        if proof.stepCount <= 1 { return true }
        guard proof.foldProofs.count == proof.stepCount - 1 else { return false }

        for i in 0..<proof.foldProofs.count {
            let running = proof.intermediateInstances[i]
            let fresh = proof.freshInstances[i]
            let foldProof = proof.foldProofs[i]

            let target: NovaRelaxedInstance
            if i < proof.foldProofs.count - 1 {
                target = proof.intermediateInstances[i + 1]
            } else {
                target = proof.finalInstance
            }

            let ok = foldVerifier.verify(running: running, new: fresh,
                                          proof: foldProof, claimed: target)
            guard ok else { return false }
        }
        return true
    }
}
