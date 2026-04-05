// DataParallelProver — Proves N instances of the same circuit efficiently
//
// Batched GKR: builds a combined circuit from N copies of the template, then runs
// the standard GKR sparse sumcheck on the combined circuit. The combined circuit
// has structured, repetitive wiring — each instance's gates only reference inputs
// from the same instance, so the wiring has N * |gates_per_layer| nonzero entries
// (not N^2). The GKR engine's sparse wiring representation naturally exploits this.
//
// Cost: O(d * N * |C_layer| * log(N * |C_layer|)) total prover work
// Proof size: O(d * log(N * |C|)) — same as a single instance proof (just bigger circuit)
//
// The key optimization over proving N instances separately:
// - Single proof, single verifier check
// - The wiring MLE has the same sparse structure repeated N times
// - GKREngine pre-computes wiring topology once for the combined circuit

import Foundation

// MARK: - Proof Types

/// Proof for the data-parallel GKR protocol.
/// Uses standard GKR layer proofs since the combined circuit is a standard layered circuit.
public struct DataParallelGKRProof {
    public let gkrProof: GKRProof
    public let allOutputs: [[Fr]]   // per-instance outputs

    public init(gkrProof: GKRProof, allOutputs: [[Fr]]) {
        self.gkrProof = gkrProof
        self.allOutputs = allOutputs
    }
}

// MARK: - DataParallelProver

public class DataParallelProver {
    public static let version = Versions.dataParallel

    public init() {}

    /// Prove all N instances of the data-parallel circuit.
    /// Builds a combined circuit and runs standard GKR on it.
    ///
    /// The combined circuit has N * |C_layer| gates per layer. Instance i's gates
    /// are at offsets [i * paddedLayerSize .. (i+1) * paddedLayerSize) and reference
    /// inputs only from instance i. This structured wiring is automatically
    /// exploited by GKREngine's sparse wiring representation.
    public func prove(circuit: inout DataParallelCircuit, transcript: Transcript) -> DataParallelGKRProof {
        // Build the combined circuit from template + instances
        let combined = circuit.buildCombinedCircuit()
        let combinedInputs = circuit.buildCombinedInputs()

        // Run standard GKR on the combined circuit
        let engine = GKREngine(circuit: combined)
        let gkrProof = engine.prove(inputs: combinedInputs, transcript: transcript)

        // Extract per-instance outputs
        let d = circuit.template.depth
        let allValues = combined.evaluate(inputs: combinedInputs)
        let outputValues = allValues[d]
        let outputLayerSize = circuit.template.layers[d - 1].paddedSize
        var allOutputs = [[Fr]]()
        allOutputs.reserveCapacity(circuit.instances)
        for i in 0..<circuit.instances {
            let start = i * outputLayerSize
            let end = min(start + outputLayerSize, outputValues.count)
            allOutputs.append(Array(outputValues[start..<end]))
        }

        return DataParallelGKRProof(gkrProof: gkrProof, allOutputs: allOutputs)
    }
}
