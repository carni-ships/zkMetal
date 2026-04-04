// UniformCircuit — Data-parallel circuit representation
// Represents N parallel evaluations of the same sub-circuit with different inputs.
// Enables O(|C| + N log N) proving instead of O(|C| * N) by exploiting
// the structured wiring predicate that factors into circuit structure x instance selection.

import Foundation

// MARK: - Sub-Circuit

/// A sub-circuit that will be evaluated N times with different inputs.
/// The sub-circuit is a standard layered arithmetic circuit.
public struct SubCircuit {
    public let layers: [CircuitLayer]
    public let inputSize: Int   // number of field elements per instance input
    public let outputSize: Int  // number of field elements per instance output

    public var depth: Int { layers.count }

    public init(layers: [CircuitLayer], inputSize: Int, outputSize: Int) {
        precondition(!layers.isEmpty, "SubCircuit must have at least one layer")
        self.layers = layers
        self.inputSize = inputSize
        self.outputSize = outputSize
    }

    /// Build a LayeredCircuit from this sub-circuit (single instance).
    public func toLayeredCircuit() -> LayeredCircuit {
        LayeredCircuit(layers: layers)
    }

    /// Evaluate the sub-circuit on a single input set.
    public func evaluate(inputs: [Fr]) -> [Fr] {
        let circuit = toLayeredCircuit()
        return circuit.evaluateOutput(inputs: inputs)
    }
}

// MARK: - Uniform Circuit

/// N parallel evaluations of the same sub-circuit.
/// The "uniform" structure means all N instances share identical wiring;
/// only the input data differs across instances.
public struct UniformCircuit {
    public let subCircuit: SubCircuit
    public let numInstances: Int       // N
    public let inputs: [[Fr]]          // N sets of inputs

    /// Number of bits to index instances: ceil(log2(N))
    public var instanceBits: Int {
        numInstances <= 1 ? 0 : Int(ceil(log2(Double(numInstances))))
    }

    /// Padded number of instances (power of 2)
    public var paddedInstances: Int {
        1 << instanceBits
    }

    public init(subCircuit: SubCircuit, inputs: [[Fr]]) {
        precondition(!inputs.isEmpty, "Must have at least one instance")
        for inp in inputs {
            precondition(inp.count == subCircuit.inputSize,
                        "Each input must have \(subCircuit.inputSize) elements")
        }
        self.subCircuit = subCircuit
        self.numInstances = inputs.count
        self.inputs = inputs
    }

    /// Evaluate all N instances, returning N output vectors.
    public func evaluateAll() -> [[Fr]] {
        inputs.map { subCircuit.evaluate(inputs: $0) }
    }

    /// Evaluate all instances and return the full layer values for each instance.
    /// allValues[instance][layer] gives the values at that layer for that instance.
    public func evaluateAllLayers() -> [[[Fr]]] {
        let circuit = subCircuit.toLayeredCircuit()
        return inputs.map { circuit.evaluate(inputs: $0) }
    }

    /// Build the "flattened" combined values for a given layer across all instances.
    /// The combined array interleaves: for each gate index g, all N instance values for g
    /// are contiguous. Total size = paddedInstances * layerSize.
    ///
    /// Layout: combined[instance * layerSize + gate] = value[instance][gate]
    /// This layout enables the eq polynomial factoring in sumcheck.
    public func combinedLayerValues(layerValues: [[Fr]], layerSize: Int) -> [Fr] {
        let padN = paddedInstances
        let totalSize = padN * layerSize
        var combined = [Fr](repeating: Fr.zero, count: totalSize)
        for inst in 0..<numInstances {
            let vals = layerValues[inst]
            for g in 0..<min(vals.count, layerSize) {
                combined[inst * layerSize + g] = vals[g]
            }
        }
        return combined
    }
}

// MARK: - Sub-Circuit Builders

extension SubCircuit {
    /// Build a simplified "hash-like" sub-circuit: alternating add/mul rounds.
    /// Each round has `width` gates that mix adjacent values.
    /// This models repeated hash evaluations (e.g., Poseidon rounds).
    public static func hashLike(logWidth: Int, numRounds: Int) -> SubCircuit {
        let width = 1 << logWidth
        precondition(width >= 2 && numRounds >= 1)

        var layers = [CircuitLayer]()
        for round in 0..<numRounds {
            var gates = [Gate]()
            gates.reserveCapacity(width)
            for j in 0..<width {
                let left = j
                let right = (j + 1) % width
                // Alternate add and mul layers for mixing
                let type: GateType = (round % 2 == 0) ? .mul : .add
                gates.append(Gate(type: type, leftInput: left, rightInput: right))
            }
            layers.append(CircuitLayer(gates: gates))
        }
        return SubCircuit(layers: layers, inputSize: width, outputSize: width)
    }

    /// Build a sub-circuit that computes x^(2^k) via repeated squaring.
    /// Input: 1 element. Output: 1 element (the result of k squarings).
    /// Models a simple "power" computation repeated across many inputs.
    public static func repeatedSquaring(rounds: Int) -> SubCircuit {
        precondition(rounds >= 1)
        var layers = [CircuitLayer]()
        for _ in 0..<rounds {
            // Single gate: output = input * input (squaring)
            let gate = Gate(type: .mul, leftInput: 0, rightInput: 0)
            layers.append(CircuitLayer(gates: [gate]))
        }
        return SubCircuit(layers: layers, inputSize: 1, outputSize: 1)
    }
}
