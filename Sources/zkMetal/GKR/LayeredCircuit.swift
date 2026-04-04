// GKR Layered Circuit Representation
// Defines arithmetic circuits as sequences of layers where each gate in layer i
// takes inputs from layer i-1. Supports add and mul gates with multilinear
// extension computation for wiring predicates.

import Foundation

// MARK: - Gate Types

public enum GateType {
    case add
    case mul
}

public struct Gate {
    public let type: GateType
    public let leftInput: Int   // index in previous layer
    public let rightInput: Int  // index in previous layer

    public init(type: GateType, leftInput: Int, rightInput: Int) {
        self.type = type
        self.leftInput = leftInput
        self.rightInput = rightInput
    }
}

public struct CircuitLayer {
    public let gates: [Gate]
    public var size: Int { gates.count }
    /// log2 of the layer size (number of variables to address gates)
    public var numVars: Int { size <= 1 ? (size == 0 ? 0 : 0) : Int(ceil(log2(Double(size)))) }

    /// Padded size = 2^numVars (circuit layers padded to power of 2)
    public var paddedSize: Int { max(1, 1 << numVars) }

    public init(gates: [Gate]) {
        self.gates = gates
    }
}

// MARK: - Layered Circuit

public struct LayeredCircuit {
    /// layers[0] = closest to input (its gates read from the actual inputs)
    /// layers[depth-1] = output layer
    public let layers: [CircuitLayer]
    public var depth: Int { layers.count }

    public init(layers: [CircuitLayer]) {
        precondition(!layers.isEmpty, "Circuit must have at least one layer")
        self.layers = layers
    }

    /// Evaluate the circuit on given inputs, returning values at every layer.
    /// layerValues[i] has the outputs of layers[i]. The "input" to layers[0] is `inputs`.
    public func evaluate(inputs: [Fr]) -> [[Fr]] {
        var layerValues = [[Fr]]()
        layerValues.reserveCapacity(depth + 1)
        layerValues.append(inputs)  // "layer -1" = inputs

        for layer in layers {
            var outputs = [Fr]()
            outputs.reserveCapacity(layer.size)
            let prev = layerValues.last!
            for gate in layer.gates {
                let l = gate.leftInput < prev.count ? prev[gate.leftInput] : Fr.zero
                let r = gate.rightInput < prev.count ? prev[gate.rightInput] : Fr.zero
                switch gate.type {
                case .add:
                    outputs.append(frAdd(l, r))
                case .mul:
                    outputs.append(frMul(l, r))
                }
            }
            layerValues.append(outputs)
        }
        return layerValues
    }

    /// Evaluate and return only the final output layer values.
    public func evaluateOutput(inputs: [Fr]) -> [Fr] {
        return evaluate(inputs: inputs).last!
    }
}

// MARK: - Multilinear Polynomial (dense, evaluation form)

/// Multilinear polynomial over {0,1}^n -> Fr, stored as evaluations on the boolean hypercube.
/// Index i corresponds to the point whose binary representation is i (MSB = variable 0).
public struct MultilinearPoly {
    public let numVars: Int
    public var evals: [Fr]

    public var size: Int { evals.count }

    public init(numVars: Int, evals: [Fr]) {
        precondition(evals.count == (1 << numVars), "Expected 2^\(numVars) evaluations, got \(evals.count)")
        self.numVars = numVars
        self.evals = evals
    }

    /// Create from a function defined on fewer indices, zero-padded to 2^numVars.
    public init(numVars: Int, values: [Fr]) {
        let n = 1 << numVars
        var padded = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<min(values.count, n) {
            padded[i] = values[i]
        }
        self.numVars = numVars
        self.evals = padded
    }

    /// Evaluate the multilinear polynomial at an arbitrary point using the standard
    /// multilinear interpolation formula:
    ///   f(r_0, ..., r_{n-1}) = Σ_{x ∈ {0,1}^n} f(x) · Π_i ((1-r_i)(1-x_i) + r_i·x_i)
    public func evaluate(at point: [Fr]) -> Fr {
        precondition(point.count == numVars)
        if numVars == 0 { return evals[0] }

        // Iterative reduction: fix variables one by one from MSB (variable 0)
        var current = evals
        for i in 0..<numVars {
            let half = current.count / 2
            let ri = point[i]
            let oneMinusRi = frSub(Fr.one, ri)
            var next = [Fr](repeating: Fr.zero, count: half)
            for j in 0..<half {
                // next[j] = (1 - r_i) * current[j] + r_i * current[j + half]
                next[j] = frAdd(frMul(oneMinusRi, current[j]), frMul(ri, current[j + half]))
            }
            current = next
        }
        return current[0]
    }

    /// Fix one variable (the MSB, variable 0) to a given value, returning a polynomial
    /// with numVars-1 variables.
    public func fixVariable(_ value: Fr) -> MultilinearPoly {
        precondition(numVars > 0)
        let half = size / 2
        let oneMinusV = frSub(Fr.one, value)
        var result = [Fr](repeating: Fr.zero, count: half)
        for j in 0..<half {
            result[j] = frAdd(frMul(oneMinusV, evals[j]), frMul(value, evals[j + half]))
        }
        return MultilinearPoly(numVars: numVars - 1, evals: result)
    }

    /// Compute the "eq" polynomial: eq(r, x) = Π_i (r_i * x_i + (1 - r_i)(1 - x_i))
    /// Returns evaluations over {0,1}^n for fixed r.
    public static func eqPoly(point: [Fr]) -> [Fr] {
        let n = point.count
        let size = 1 << n
        var eq = [Fr](repeating: Fr.zero, count: size)
        eq[0] = Fr.one

        for i in 0..<n {
            let half = 1 << i
            let ri = point[i]
            let oneMinusRi = frSub(Fr.one, ri)
            // Process in reverse to avoid overwriting values we still need
            for j in stride(from: half - 1, through: 0, by: -1) {
                eq[2 * j + 1] = frMul(eq[j], ri)
                eq[2 * j] = frMul(eq[j], oneMinusRi)
            }
        }
        return eq
    }
}

// MARK: - Wiring Predicate MLEs

extension LayeredCircuit {
    /// Number of variables needed to index gates in layer i (0-indexed, i=0 is first compute layer).
    public func outputVars(layer i: Int) -> Int {
        layers[i].numVars
    }

    /// Number of variables needed to index gates/values in the input to layer i.
    /// For layer 0, this is the number of input variables.
    public func inputVars(layer i: Int) -> Int {
        if i == 0 {
            // The input to layer 0 is the circuit input.
            // We need to know the input size — infer from max gate indices.
            var maxIdx = 0
            for gate in layers[0].gates {
                maxIdx = max(maxIdx, gate.leftInput, gate.rightInput)
            }
            let inputSize = maxIdx + 1
            return inputSize <= 1 ? (inputSize == 0 ? 0 : 0) : Int(ceil(log2(Double(inputSize))))
        } else {
            return layers[i - 1].numVars
        }
    }

    /// Build the multilinear extension of the add wiring predicate for layer i.
    /// add_i(z, x, y) = 1 iff gate z in layer i is an add gate with inputs x, y.
    /// Variables: outputVars(i) + inputVars(i) + inputVars(i)
    public func addMLEForLayer(_ i: Int) -> MultilinearPoly {
        let nOut = outputVars(layer: i)
        let nIn = inputVars(layer: i)
        let totalVars = nOut + 2 * nIn
        let totalSize = 1 << totalVars
        let inSize = 1 << nIn

        var evals = [Fr](repeating: Fr.zero, count: totalSize)
        for (gIdx, gate) in layers[i].gates.enumerated() {
            guard gate.type == .add else { continue }
            let lIdx = gate.leftInput
            let rIdx = gate.rightInput
            // Index = gIdx * inSize^2 + lIdx * inSize + rIdx
            let idx = gIdx * inSize * inSize + lIdx * inSize + rIdx
            if idx < totalSize {
                evals[idx] = Fr.one
            }
        }
        return MultilinearPoly(numVars: totalVars, evals: evals)
    }

    /// Build the multilinear extension of the mul wiring predicate for layer i.
    public func mulMLEForLayer(_ i: Int) -> MultilinearPoly {
        let nOut = outputVars(layer: i)
        let nIn = inputVars(layer: i)
        let totalVars = nOut + 2 * nIn
        let totalSize = 1 << totalVars
        let inSize = 1 << nIn

        var evals = [Fr](repeating: Fr.zero, count: totalSize)
        for (gIdx, gate) in layers[i].gates.enumerated() {
            guard gate.type == .mul else { continue }
            let lIdx = gate.leftInput
            let rIdx = gate.rightInput
            let idx = gIdx * inSize * inSize + lIdx * inSize + rIdx
            if idx < totalSize {
                evals[idx] = Fr.one
            }
        }
        return MultilinearPoly(numVars: totalVars, evals: evals)
    }
}

// MARK: - Circuit Builders

extension LayeredCircuit {
    /// Build a "hash-like" circuit: each layer has width gates, half add + half mul,
    /// with interleaved wiring that mixes adjacent values.
    public static func repeatedHashCircuit(logWidth: Int, depth: Int) -> LayeredCircuit {
        let width = 1 << logWidth
        precondition(width >= 2 && depth >= 1)

        var layers = [CircuitLayer]()
        for _ in 0..<depth {
            var gates = [Gate]()
            gates.reserveCapacity(width)
            for j in 0..<width {
                let left = j
                let right = (j + 1) % width
                let type: GateType = (j < width / 2) ? .add : .mul
                gates.append(Gate(type: type, leftInput: left, rightInput: right))
            }
            layers.append(CircuitLayer(gates: gates))
        }
        return LayeredCircuit(layers: layers)
    }

    /// Build a simple depth-1 inner product circuit:
    /// Layer 0: n mul gates computing a_i * b_i
    /// Layer 1: tree of add gates reducing to a single sum
    public static func innerProductCircuit(size: Int) -> LayeredCircuit {
        precondition(size >= 2 && (size & (size - 1)) == 0, "Size must be power of 2")
        // Layer 0: mul gates, gate i takes inputs 2i and 2i+1
        var mulGates = [Gate]()
        for i in 0..<size {
            mulGates.append(Gate(type: .mul, leftInput: 2 * i, rightInput: 2 * i + 1))
        }
        var layers = [CircuitLayer(gates: mulGates)]

        // Add tree layers
        var currentSize = size
        while currentSize > 1 {
            var addGates = [Gate]()
            let nextSize = currentSize / 2
            for i in 0..<nextSize {
                addGates.append(Gate(type: .add, leftInput: 2 * i, rightInput: 2 * i + 1))
            }
            layers.append(CircuitLayer(gates: addGates))
            currentSize = nextSize
        }

        return LayeredCircuit(layers: layers)
    }
}
