// ProtogalaxyIVC -- Full Incremental Verifiable Computation pipeline
//
// Combines Protogalaxy folding with the sumcheck-based decider to provide
// a complete IVC system:
//   1. step():  Fold one more computation step into the running accumulator
//   2. prove(): Produce a final SNARK proof from the accumulated state
//   3. verify(): Verify the final proof without access to witnesses
//
// Usage:
//   let ivc = ProtogalaxyIVC(circuitSize: 4)
//   for (instance, witness) in computationSteps {
//       ivc.step(instance: instance, witness: witness)
//   }
//   let proof = ivc.prove()
//   let valid = ProtogalaxyIVC.verify(proof: proof)
//
// The accumulator maintains:
//   - A running ProtogalaxyInstance (commitments, challenges, error, u)
//   - Running witness polynomials (folded a, b, c columns)
//   - The chain of folding proofs (for full chain verification)
//   - The original instances (for IVC chain verification)
//
// Reference: "ProtoGalaxy: Efficient ProtoStar-style folding of multiple instances"
//            (Gabizon, Khovratovich 2023)

import Foundation
import NeonFieldOps

// MARK: - IVC State

/// The running state of the Protogalaxy IVC pipeline.
public struct ProtogalaxyIVCState {
    /// The current accumulated instance
    public let instance: ProtogalaxyInstance
    /// The current accumulated witness columns
    public let witnesses: [[Fr]]
    /// Number of computation steps folded so far
    public let stepCount: Int
}

// MARK: - Protogalaxy IVC

/// Full IVC pipeline combining Protogalaxy folding with the sumcheck decider.
///
/// Maintains a running accumulator across computation steps. Each step folds
/// a new Plonk instance into the accumulator. At any point, `prove()` produces
/// a compact proof that the entire chain of computations was valid.
///
/// Thread safety: not thread-safe. Designed for single-threaded sequential folding.
public class ProtogalaxyIVC {
    /// Circuit parameters
    public let circuitSize: Int
    public let numWitnessColumns: Int

    /// Internal prover for folding steps
    private let folder: ProtogalaxyProver

    /// Running accumulated state (nil before first step)
    private var runningInstance: ProtogalaxyInstance?
    private var runningWitnesses: [[Fr]]?

    /// History for full chain verification
    private var originalInstances: [ProtogalaxyInstance]
    private var foldingProofs: [ProtogalaxyFoldingProof]
    private var stepCount: Int

    /// Initialize the IVC pipeline.
    ///
    /// - Parameters:
    ///   - circuitSize: Number of gates (must be power of 2)
    ///   - numWitnessColumns: Number of witness columns (default 3: a, b, c)
    public init(circuitSize: Int, numWitnessColumns: Int = 3) {
        precondition(circuitSize > 0 && (circuitSize & (circuitSize - 1)) == 0,
                     "Circuit size must be a power of 2")
        self.circuitSize = circuitSize
        self.numWitnessColumns = numWitnessColumns
        self.folder = ProtogalaxyProver(circuitSize: circuitSize,
                                         numWitnessColumns: numWitnessColumns)
        self.runningInstance = nil
        self.runningWitnesses = nil
        self.originalInstances = []
        self.foldingProofs = []
        self.stepCount = 0
    }

    // MARK: - Step

    /// Fold one more computation step into the running accumulator.
    ///
    /// The first step initializes the accumulator. Subsequent steps fold
    /// the new instance into the running accumulated instance.
    ///
    /// - Parameters:
    ///   - instance: A fresh Plonk instance for this computation step
    ///   - witness: Witness polynomials [a_evals, b_evals, c_evals]
    /// - Returns: The current accumulated state after this step
    @discardableResult
    public func step(instance: ProtogalaxyInstance,
                     witness: [[Fr]]) -> ProtogalaxyIVCState {
        precondition(witness.count == numWitnessColumns,
                     "Expected \(numWitnessColumns) witness columns")
        for col in witness {
            precondition(col.count == circuitSize,
                         "Witness column size \(col.count) != circuit size \(circuitSize)")
        }

        originalInstances.append(instance)

        if runningInstance == nil {
            // First step: initialize the accumulator
            runningInstance = instance
            runningWitnesses = witness
            stepCount = 1
        } else {
            // Fold the new instance into the running accumulator
            let (folded, foldedWit, proof) = folder.fold(
                instances: [runningInstance!, instance],
                witnesses: [runningWitnesses!, witness]
            )
            runningInstance = folded
            runningWitnesses = foldedWit
            foldingProofs.append(proof)
            stepCount += 1
        }

        return ProtogalaxyIVCState(
            instance: runningInstance!,
            witnesses: runningWitnesses!,
            stepCount: stepCount
        )
    }

    // MARK: - Prove

    /// Produce a final SNARK proof from the current accumulated state.
    ///
    /// This runs the Spartan-style sumcheck decider on the accumulated
    /// instance to produce a compact proof. The proof includes the folding
    /// chain so the verifier can check the entire IVC computation.
    ///
    /// - Returns: A ProtogalaxyDeciderProof that can be verified without witnesses
    public func prove() -> ProtogalaxyDeciderProof {
        guard let instance = runningInstance, let witnesses = runningWitnesses else {
            preconditionFailure("Cannot prove: no steps have been performed")
        }

        let config = ProtogalaxyDeciderConfig(circuitSize: circuitSize,
                                                numWitnessColumns: numWitnessColumns)
        let decider = ProtogalaxyDeciderProver(config: config)
        return decider.decide(instance: instance, witnesses: witnesses,
                              foldingProofs: foldingProofs)
    }

    // MARK: - Verify

    /// Verify a decider proof (static, no state needed).
    ///
    /// Checks:
    ///   1. The sumcheck proof is valid
    ///   2. The witness evaluations are consistent
    ///
    /// - Parameter proof: The decider proof to verify
    /// - Returns: true if the proof is valid
    public static func verify(proof: ProtogalaxyDeciderProof) -> Bool {
        let verifier = ProtogalaxyDeciderVerifier()
        return verifier.verify(proof: proof)
    }

    /// Verify a decider proof with full IVC chain verification.
    ///
    /// In addition to the sumcheck proof, verifies that each folding step
    /// in the chain was performed correctly.
    ///
    /// - Parameters:
    ///   - proof: The decider proof with folding proofs
    ///   - originalInstances: The original instances that were folded
    /// - Returns: true if the entire chain and final proof are valid
    public static func verifyIVC(proof: ProtogalaxyDeciderProof,
                                  originalInstances: [ProtogalaxyInstance]) -> Bool {
        let verifier = ProtogalaxyDeciderVerifier()
        return verifier.verifyIVCChain(proof: proof, originalInstances: originalInstances)
    }

    // MARK: - State Access

    /// Current accumulated state (nil if no steps performed).
    public var currentState: ProtogalaxyIVCState? {
        guard let inst = runningInstance, let wit = runningWitnesses else { return nil }
        return ProtogalaxyIVCState(instance: inst, witnesses: wit, stepCount: stepCount)
    }

    /// Number of steps folded so far.
    public var currentStepCount: Int { stepCount }

    /// The chain of original instances folded so far.
    public var instanceHistory: [ProtogalaxyInstance] { originalInstances }

    // MARK: - Reset

    /// Reset the IVC pipeline to its initial state.
    public func reset() {
        runningInstance = nil
        runningWitnesses = nil
        originalInstances = []
        foldingProofs = []
        stepCount = 0
    }
}
