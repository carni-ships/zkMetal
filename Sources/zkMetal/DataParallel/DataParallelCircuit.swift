// DataParallelCircuit — N copies of the same LayeredCircuit with different inputs
//
// Key insight: when the same sub-circuit (wiring pattern) is evaluated N times,
// the wiring MLE factors into:
//   W_combined(z_inst, z_circ, x_inst, x_circ, y_inst, y_circ) =
//     eq(z_inst, x_inst) * eq(x_inst, y_inst) * W_sub(z_circ, x_circ, y_circ)
//
// This means we compute the wiring MLE once for the template circuit and amortize
// it across all N instances. The prover work is O(N * |C|) where |C| is the
// sub-circuit size, vs O(N^2 * |C|) for a flat constraint system.

import Foundation

// MARK: - DataParallelCircuit

/// Represents N parallel evaluations of the same LayeredCircuit template.
/// Stores the shared wiring topology once and separate I/O per instance.
public struct DataParallelCircuit {
    /// The template circuit — all N instances share this wiring topology.
    public let template: LayeredCircuit

    /// Number of parallel instances.
    public let instances: Int

    /// Per-instance inputs. inputs[i] is the input vector for instance i.
    public private(set) var instanceInputs: [[Fr]]

    /// Per-instance outputs (populated after evaluation).
    public private(set) var instanceOutputs: [[Fr]]?

    /// Full layer values per instance (populated after evaluation).
    /// allLayerValues[instance][layerIdx] gives values at that layer.
    public private(set) var allLayerValues: [[[Fr]]]?

    /// Number of bits to index instances: ceil(log2(N)), at least 1 for N >= 2.
    public var instanceBits: Int {
        instances <= 1 ? 0 : Int(ceil(log2(Double(instances))))
    }

    /// Padded instance count (next power of 2).
    public var paddedInstances: Int {
        1 << instanceBits
    }

    /// Sub-circuit input size (max gate input index + 1 in layer 0).
    public var inputSize: Int {
        var maxIdx = 0
        if !template.layers.isEmpty {
            for g in template.layers[0].gates {
                maxIdx = max(maxIdx, g.leftInput, g.rightInput)
            }
        }
        return maxIdx + 1
    }

    /// Sub-circuit output size.
    public var outputSize: Int {
        template.layers.isEmpty ? 0 : template.layers.last!.size
    }

    // MARK: - Cached Wiring MLEs

    /// Cached add wiring MLEs per layer (computed once, shared across instances).
    private var _cachedAddMLEs: [MultilinearPoly?]
    /// Cached mul wiring MLEs per layer.
    private var _cachedMulMLEs: [MultilinearPoly?]

    // MARK: - Init

    /// Create a data-parallel circuit from a template and N instance inputs.
    /// - Parameters:
    ///   - template: The shared LayeredCircuit defining the wiring topology.
    ///   - instances: Number of parallel evaluations.
    ///   - inputs: Optional per-instance inputs. If nil, must be set before proving.
    public init(template: LayeredCircuit, instances: Int, inputs: [[Fr]]? = nil) {
        precondition(instances >= 1, "Need at least 1 instance")
        self.template = template
        self.instances = instances
        self.instanceInputs = inputs ?? []
        self.instanceOutputs = nil
        self.allLayerValues = nil
        self._cachedAddMLEs = [MultilinearPoly?](repeating: nil, count: template.depth)
        self._cachedMulMLEs = [MultilinearPoly?](repeating: nil, count: template.depth)
    }

    /// Set inputs for all instances.
    public mutating func setInputs(_ inputs: [[Fr]]) {
        precondition(inputs.count == instances,
                     "Expected \(instances) input sets, got \(inputs.count)")
        self.instanceInputs = inputs
        self.instanceOutputs = nil
        self.allLayerValues = nil
    }

    // MARK: - Evaluation

    /// Evaluate all N instances, populating outputs and layer values.
    @discardableResult
    public mutating func evaluateAll() -> [[Fr]] {
        precondition(instanceInputs.count == instances,
                     "Inputs not set: have \(instanceInputs.count), need \(instances)")

        var layerVals = [[[Fr]]]()
        layerVals.reserveCapacity(instances)
        var outputs = [[Fr]]()
        outputs.reserveCapacity(instances)

        for i in 0..<instances {
            let vals = template.evaluate(inputs: instanceInputs[i])
            outputs.append(vals.last!)
            layerVals.append(vals)
        }

        self.allLayerValues = layerVals
        self.instanceOutputs = outputs
        return outputs
    }

    // MARK: - Combined MLE Construction

    /// Build combined layer values across all instances for a given layer.
    /// Layout: combined[instance * layerPadSize + gateIdx] = value[instance][gateIdx]
    /// This layout enables eq polynomial factoring in sumcheck.
    public func combinedValues(layerIndex: Int) -> [Fr] {
        guard let allVals = allLayerValues else {
            preconditionFailure("Must call evaluateAll() before combinedValues()")
        }
        let layerSize: Int
        if layerIndex == 0 {
            // Input layer
            layerSize = 1 << inputVarsForLayer(0)
        } else {
            layerSize = template.layers[layerIndex - 1].paddedSize
        }
        return buildCombined(values: allVals.map { $0[layerIndex] }, padSize: layerSize)
    }

    /// Build combined output values.
    public func combinedOutputValues() -> [Fr] {
        guard let allVals = allLayerValues else {
            preconditionFailure("Must call evaluateAll() before combinedOutputValues()")
        }
        let d = template.depth
        let outputPadSize = template.layers[d - 1].paddedSize
        return buildCombined(values: allVals.map { $0[d] }, padSize: outputPadSize)
    }

    /// Build combined input values.
    public func combinedInputValues() -> [Fr] {
        guard let allVals = allLayerValues else {
            preconditionFailure("Must call evaluateAll() before combinedInputValues()")
        }
        let inputPadSize = 1 << inputVarsForLayer(0)
        return buildCombined(values: allVals.map { $0[0] }, padSize: inputPadSize)
    }

    private func buildCombined(values: [[Fr]], padSize: Int) -> [Fr] {
        let padN = paddedInstances
        var combined = [Fr](repeating: Fr.zero, count: padN * padSize)
        for (inst, vals) in values.enumerated() {
            for (g, v) in vals.prefix(padSize).enumerated() {
                combined[inst * padSize + g] = v
            }
        }
        return combined
    }

    // MARK: - Wiring MLEs (Shared)

    /// Get or build the add wiring MLE for the given layer.
    /// This is computed once from the template and reused across all instances.
    public mutating func addWiringMLE(layer: Int) -> MultilinearPoly {
        if let cached = _cachedAddMLEs[layer] { return cached }
        let mle = buildWiringMLE(layerIdx: layer, type: .add)
        _cachedAddMLEs[layer] = mle
        return mle
    }

    /// Get or build the mul wiring MLE for the given layer.
    public mutating func mulWiringMLE(layer: Int) -> MultilinearPoly {
        if let cached = _cachedMulMLEs[layer] { return cached }
        let mle = buildWiringMLE(layerIdx: layer, type: .mul)
        _cachedMulMLEs[layer] = mle
        return mle
    }

    /// Build wiring MLE for a single layer and gate type from the template circuit.
    /// Variables: (z_circ, x_circ, y_circ) where z indexes output gates, x/y index inputs.
    private func buildWiringMLE(layerIdx: Int, type: GateType) -> MultilinearPoly {
        let nOut = template.outputVars(layer: layerIdx)
        let nIn = inputVarsForLayer(layerIdx)
        let totalVars = nOut + 2 * nIn
        let totalSize = 1 << totalVars
        let inSize = 1 << nIn

        var evals = [Fr](repeating: Fr.zero, count: totalSize)
        for (gIdx, gate) in template.layers[layerIdx].gates.enumerated() {
            guard gate.type == type else { continue }
            let idx = gIdx * inSize * inSize + gate.leftInput * inSize + gate.rightInput
            if idx < totalSize {
                evals[idx] = Fr.one
            }
        }
        return MultilinearPoly(numVars: totalVars, evals: evals)
    }

    // MARK: - Variable Counts

    /// Number of variables to index gates in the input to the given layer.
    public func inputVarsForLayer(_ layerIdx: Int) -> Int {
        if layerIdx == 0 {
            let sz = inputSize
            return sz <= 1 ? 0 : Int(ceil(log2(Double(sz))))
        } else {
            return template.layers[layerIdx - 1].numVars
        }
    }

    /// Number of output variables for the given layer.
    public func outputVarsForLayer(_ layerIdx: Int) -> Int {
        template.outputVars(layer: layerIdx)
    }
}
