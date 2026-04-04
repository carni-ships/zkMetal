// DataParallelEngine -- O(|C| * N + N log N) proof for N repetitions of the same sub-circuit
//
// Core insight (C21): When a circuit has N identical copies of a sub-circuit C, the wiring
// pattern repeats. We build a combined circuit with N * |C| gates per layer where instance i
// uses gates at offsets [i * |C|_layer .. (i+1) * |C|_layer). Each gate within instance i
// references inputs from instance i only, preserving independence.
//
// The GKR sparse sumcheck exploits the repetitive structure:
//   - Wiring has N * |gates_per_layer| nonzero entries (not N^2)
//   - Per-layer cost: O(N * |C_layer|) for the sparse wiring iteration
//   - Total cost: O(N * |C| + N * |C| * log(N * |C|)) per layer
//
// For N instances of a circuit with |C| gates and depth d, this gives:
//   O(d * N * |C| * log(N * |C|)) total prover work
// vs O(N * d * |C| * log(|C|)) for N independent proofs.
// The key savings: single proof, single verifier check, proof size = O(d * log(N * |C|)).

import Foundation

// MARK: - Proof Types

public struct DataParallelProof {
    public let layerProofs: [GKRLayerProof]
    public let allOutputs: [[Fr]]

    public init(layerProofs: [GKRLayerProof], allOutputs: [[Fr]]) {
        self.layerProofs = layerProofs
        self.allOutputs = allOutputs
    }
}

// MARK: - Engine

public class DataParallelEngine {
    public static let version = Versions.dataParallel

    public init() {}

    // MARK: - Prover

    /// Prove N instances of the same sub-circuit using a single combined GKR proof.
    /// The combined circuit has N * |C_layer| gates per layer with structured wiring.
    public func prove(circuit: UniformCircuit, transcript: Transcript) -> DataParallelProof {
        let combined = circuit.buildCombinedCircuit()
        let combinedInputs = circuit.buildCombinedInputs()
        let engine = GKREngine(circuit: combined)
        let proof = engine.prove(inputs: combinedInputs, transcript: transcript)

        // Extract per-instance outputs
        let sub = circuit.subCircuit
        let d = sub.depth
        let allValues = combined.evaluate(inputs: combinedInputs)
        let outputValues = allValues[d]
        let outputLayerSize = sub.layers[d - 1].paddedSize
        var allOutputs = [[Fr]]()
        for i in 0..<circuit.numInstances {
            let start = i * outputLayerSize
            let end = min(start + outputLayerSize, outputValues.count)
            allOutputs.append(Array(outputValues[start..<end]))
        }

        return DataParallelProof(layerProofs: proof.layerProofs, allOutputs: allOutputs)
    }

    // MARK: - Verifier

    /// Verify a data-parallel proof.
    public func verify(
        subCircuit: SubCircuit, numInstances: Int,
        inputs: [[Fr]], proof: DataParallelProof,
        transcript: Transcript
    ) -> Bool {
        let uniformCircuit = UniformCircuit(subCircuit: subCircuit, inputs: inputs)
        let combined = uniformCircuit.buildCombinedCircuit()
        let combinedInputs = uniformCircuit.buildCombinedInputs()
        let output = combined.evaluateOutput(inputs: combinedInputs)
        let engine = GKREngine(circuit: combined)
        let gkrProof = GKRProof(layerProofs: proof.layerProofs)
        return engine.verify(inputs: combinedInputs, output: output, proof: gkrProof, transcript: transcript)
    }
}

// MARK: - Combined Circuit Construction

extension UniformCircuit {
    /// Build a combined LayeredCircuit where instance i's gates are at offset i * layerSize.
    /// Gate (i, g) in the combined circuit references inputs from instance i only.
    func buildCombinedCircuit() -> LayeredCircuit {
        let sub = subCircuit
        let padN = paddedInstances

        var combinedLayers = [CircuitLayer]()
        for layerIdx in 0..<sub.depth {
            let subLayer = sub.layers[layerIdx]
            let subSize = subLayer.paddedSize

            // For layer 0, inputs are at offset i * inputPaddedSize
            // For layer > 0, inputs are at offset i * prevLayerPaddedSize
            let inputPadSize: Int
            if layerIdx == 0 {
                var maxIdx = 0
                for g in subLayer.gates { maxIdx = max(maxIdx, g.leftInput, g.rightInput) }
                inputPadSize = max(1, 1 << Int(ceil(log2(Double(maxIdx + 1)))))
            } else {
                inputPadSize = sub.layers[layerIdx - 1].paddedSize
            }

            var gates = [Gate]()
            gates.reserveCapacity(padN * subSize)
            for inst in 0..<padN {
                let inputOffset = inst * inputPadSize
                for g in subLayer.gates {
                    gates.append(Gate(type: g.type,
                                     leftInput: inputOffset + g.leftInput,
                                     rightInput: inputOffset + g.rightInput))
                }
                // Pad to subSize with dummy gates (add 0+0)
                for _ in subLayer.gates.count..<subSize {
                    gates.append(Gate(type: .add, leftInput: inputOffset, rightInput: inputOffset))
                }
            }
            combinedLayers.append(CircuitLayer(gates: gates))
        }
        return LayeredCircuit(layers: combinedLayers)
    }

    /// Build combined inputs: concatenate all instance inputs, padded to power of 2 per instance.
    func buildCombinedInputs() -> [Fr] {
        let sub = subCircuit
        let padN = paddedInstances

        // Determine input pad size
        let firstLayer = sub.layers[0]
        var maxIdx = 0
        for g in firstLayer.gates { maxIdx = max(maxIdx, g.leftInput, g.rightInput) }
        let inputPadSize = max(1, 1 << Int(ceil(log2(Double(maxIdx + 1)))))

        var combined = [Fr](repeating: Fr.zero, count: padN * inputPadSize)
        for (inst, inp) in inputs.enumerated() {
            for (j, v) in inp.enumerated() {
                combined[inst * inputPadSize + j] = v
            }
        }
        return combined
    }
}
