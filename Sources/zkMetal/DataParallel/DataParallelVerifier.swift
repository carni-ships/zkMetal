// DataParallelVerifier — Verifies a data-parallel GKR proof
//
// Verification cost is essentially the same as single-circuit GKR verification
// on the combined circuit, plus O(N) field ops to reconstruct the combined
// output from per-instance outputs.
//
// The verifier reconstructs the combined circuit from the template (building
// the gate structure from the shared wiring pattern) and runs standard GKR
// verification. The combined circuit's wiring is structured — each instance's
// gates reference only inputs from that instance — but the verifier doesn't
// need to special-case this; standard GKR verification handles it.

import Foundation

// MARK: - DataParallelVerifier

public class DataParallelVerifier {

    public init() {}

    /// Verify a data-parallel proof.
    ///
    /// Reconstructs the combined circuit from the template and per-instance inputs,
    /// then runs standard GKR verification.
    ///
    /// - Parameters:
    ///   - template: The shared circuit topology (same as used by prover).
    ///   - numInstances: Number of parallel instances (N).
    ///   - inputs: Per-instance input vectors.
    ///   - proof: The data-parallel proof to verify.
    ///   - transcript: Fiat-Shamir transcript (must match prover's).
    /// - Returns: true if the proof is valid.
    public func verify(
        template: LayeredCircuit,
        numInstances: Int,
        inputs: [[Fr]],
        proof: DataParallelGKRProof,
        transcript: Transcript
    ) -> Bool {
        guard proof.allOutputs.count == numInstances else { return false }

        // Reconstruct the combined circuit from the template
        let dpCircuit = DataParallelCircuit(
            template: template, instances: numInstances, inputs: inputs)
        let combined = dpCircuit.buildCombinedCircuit()
        let combinedInputs = dpCircuit.buildCombinedInputs()

        // Reconstruct combined output from per-instance outputs
        let output = combined.evaluateOutput(inputs: combinedInputs)

        // Run standard GKR verification on the combined circuit
        let engine = GKREngine(circuit: combined)
        return engine.verify(
            inputs: combinedInputs,
            output: output,
            proof: proof.gkrProof,
            transcript: transcript)
    }
}
