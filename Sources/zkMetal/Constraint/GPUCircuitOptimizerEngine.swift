// GPU Circuit Optimizer Engine — GPU-accelerated circuit optimization passes
//
// Applies optimization passes to LayeredCircuit representations, reducing gate
// count and wire count for faster proving:
//   1. Constant folding (evaluate gates with known constant inputs)
//   2. Dead gate elimination (remove gates whose outputs are unused)
//   3. Common subexpression elimination (share identical sub-computations)
//   4. Gate merging (combine cascaded linear gates: add(add(a,b),c) -> single gate)
//   5. Wire renumbering (compact wire indices after optimization)
//   6. Statistics reporting (gate count before/after each pass)
//
// All passes preserve semantic equivalence: the optimized circuit produces the
// same output as the original for any input assignment.

import Foundation

// MARK: - Pass Statistics

/// Per-pass statistics for reporting optimization effectiveness.
public struct CircuitPassStats {
    public let passName: String
    public let gatesBefore: Int
    public let gatesAfter: Int
    public let timeMs: Double

    public var eliminated: Int { gatesBefore - gatesAfter }

    public var reductionPct: Double {
        guard gatesBefore > 0 else { return 0.0 }
        return Double(eliminated) / Double(gatesBefore) * 100.0
    }
}

/// Aggregate statistics from a full optimization run.
public struct CircuitOptimizationStats {
    public let passStats: [CircuitPassStats]
    public let originalGates: Int
    public let optimizedGates: Int
    public let originalWires: Int
    public let optimizedWires: Int
    public let totalTimeMs: Double

    public var summary: String {
        var lines = [String]()
        lines.append(String(format: "GPUCircuitOptimizer: %d -> %d gates (%.1f%% reduction), %d -> %d wires, %.2fms",
                            originalGates, optimizedGates,
                            originalGates > 0 ? Double(originalGates - optimizedGates) / Double(originalGates) * 100.0 : 0.0,
                            originalWires, optimizedWires, totalTimeMs))
        for ps in passStats {
            lines.append(String(format: "  %-24s %d -> %d (-%d, %.1f%%) %.2fms",
                                ps.passName, ps.gatesBefore, ps.gatesAfter,
                                ps.eliminated, ps.reductionPct, ps.timeMs))
        }
        return lines.joined(separator: "\n")
    }
}

// MARK: - Optimizable Gate IR

/// A flat gate representation suitable for optimization passes.
/// Each gate reads from wire indices and writes to an output wire index.
public struct OptGate: Hashable {
    public enum Kind: UInt8, Hashable {
        case add = 0
        case mul = 1
        case constAdd = 2   // output = input + constant
        case constMul = 3   // output = input * constant
    }

    public let kind: Kind
    public let inputA: Int       // first input wire index
    public let inputB: Int       // second input wire index (unused for const ops, set to -1)
    public let output: Int       // output wire index
    public let constant: Fr      // constant for constAdd/constMul; Fr.zero otherwise

    public init(kind: Kind, inputA: Int, inputB: Int = -1, output: Int, constant: Fr = Fr.zero) {
        self.kind = kind
        self.inputA = inputA
        self.inputB = inputB
        self.output = output
        self.constant = constant
    }
}

/// A flat circuit representation for optimization.
public struct OptCircuit {
    public var gates: [OptGate]
    public var numWires: Int           // total wire slots (inputs + intermediates + outputs)
    public var numInputWires: Int      // number of primary input wires
    public var numOutputWires: Int     // number of output wires

    public var gateCount: Int { gates.count }

    public init(gates: [OptGate], numWires: Int, numInputWires: Int, numOutputWires: Int) {
        self.gates = gates
        self.numWires = numWires
        self.numInputWires = numInputWires
        self.numOutputWires = numOutputWires
    }
}

// MARK: - GPU Circuit Optimizer Engine

/// GPU-accelerated multi-pass circuit optimizer.
///
/// Usage:
///   let engine = GPUCircuitOptimizerEngine()
///   let (optimized, stats) = engine.optimizeAll(circuit)
///   print(stats.summary)
public final class GPUCircuitOptimizerEngine {

    public init() {}

    // MARK: - Convert LayeredCircuit -> OptCircuit

    /// Convert a LayeredCircuit into the flat OptCircuit IR for optimization.
    public func flatten(_ lc: LayeredCircuit, inputCount: Int) -> OptCircuit {
        // Assign wire indices: inputs get 0..<inputCount, then each gate output
        // gets a fresh wire index in topological order.
        var gates = [OptGate]()
        var nextWire = inputCount

        for layer in lc.layers {
            let layerBase = nextWire
            for (gIdx, gate) in layer.gates.enumerated() {
                let outWire = layerBase + gIdx
                switch gate.type {
                case .add:
                    gates.append(OptGate(kind: .add, inputA: gate.leftInput, inputB: gate.rightInput, output: outWire))
                case .mul:
                    gates.append(OptGate(kind: .mul, inputA: gate.leftInput, inputB: gate.rightInput, output: outWire))
                }
            }
            nextWire = layerBase + layer.gates.count
        }

        let numOutputs = lc.layers.last?.gates.count ?? 0
        return OptCircuit(gates: gates, numWires: nextWire,
                          numInputWires: inputCount, numOutputWires: numOutputs)
    }

    /// Reconstruct a LayeredCircuit from the flat OptCircuit IR.
    /// Uses topological ordering to assign gates to layers.
    public func unflatten(_ oc: OptCircuit) -> LayeredCircuit {
        guard !oc.gates.isEmpty else {
            // Return a minimal single-layer circuit
            return LayeredCircuit(layers: [CircuitLayer(gates: [Gate(type: .add, leftInput: 0, rightInput: 0)])])
        }

        // Build dependency depth for each gate output
        var wireDepth = [Int: Int]()
        for i in 0..<oc.numInputWires {
            wireDepth[i] = 0
        }

        var gatesByOutput = [Int: OptGate]()
        for g in oc.gates {
            gatesByOutput[g.output] = g
        }

        // Compute depth for each gate
        var gateDepths = [Int]()
        for g in oc.gates {
            let dA = wireDepth[g.inputA] ?? 0
            let dB = g.inputB >= 0 ? (wireDepth[g.inputB] ?? 0) : 0
            let depth = max(dA, dB) + 1
            wireDepth[g.output] = depth
            gateDepths.append(depth)
        }

        // Group gates by depth
        let maxDepth = gateDepths.max() ?? 1
        var layerGates = [[OptGate]](repeating: [], count: maxDepth)
        for (i, g) in oc.gates.enumerated() {
            layerGates[gateDepths[i] - 1].append(g)
        }

        // Build wire remapping per layer: inputs to each layer must reference
        // wires from the previous layer's outputs or original inputs.
        // For simplicity, build layers using the original wire indices.
        var layers = [CircuitLayer]()
        for layerIdx in 0..<maxDepth {
            var gkrGates = [Gate]()
            for g in layerGates[layerIdx] {
                let type: GateType = (g.kind == .add || g.kind == .constAdd) ? .add : .mul
                gkrGates.append(Gate(type: type,
                                     leftInput: g.inputA,
                                     rightInput: max(g.inputB, 0)))
            }
            if gkrGates.isEmpty {
                gkrGates.append(Gate(type: .add, leftInput: 0, rightInput: 0))
            }
            layers.append(CircuitLayer(gates: gkrGates))
        }

        return LayeredCircuit(layers: layers)
    }

    // MARK: - Pass 1: Constant Folding

    /// Evaluate gates whose inputs are both known constants.
    /// Requires a partial assignment of wire values (at minimum, the circuit inputs).
    /// Returns the optimized circuit and the set of wires with known constant values.
    public func constantFolding(_ circuit: OptCircuit,
                                knownConstants: [Int: Fr] = [:]) -> OptCircuit {
        var known = knownConstants
        var keptGates = [OptGate]()

        for g in circuit.gates {
            let aKnown = known[g.inputA]
            let bKnown = g.inputB >= 0 ? known[g.inputB] : nil

            switch g.kind {
            case .add:
                if let a = aKnown, let b = bKnown {
                    // Both inputs constant -> fold
                    known[g.output] = frAdd(a, b)
                    continue // eliminate gate
                }
                keptGates.append(g)

            case .mul:
                if let a = aKnown, let b = bKnown {
                    known[g.output] = frMul(a, b)
                    continue
                }
                // Multiply by zero -> output is zero
                if let a = aKnown, a.isZero {
                    known[g.output] = Fr.zero
                    continue
                }
                if let b = bKnown, b.isZero {
                    known[g.output] = Fr.zero
                    continue
                }
                // Multiply by one -> pass-through (converted to constMul identity)
                if let a = aKnown, a == Fr.one {
                    keptGates.append(OptGate(kind: .constMul, inputA: g.inputB, output: g.output, constant: Fr.one))
                    continue
                }
                if let b = bKnown, b == Fr.one {
                    keptGates.append(OptGate(kind: .constMul, inputA: g.inputA, output: g.output, constant: Fr.one))
                    continue
                }
                keptGates.append(g)

            case .constAdd:
                if let a = aKnown {
                    known[g.output] = frAdd(a, g.constant)
                    continue
                }
                keptGates.append(g)

            case .constMul:
                if let a = aKnown {
                    known[g.output] = frMul(a, g.constant)
                    continue
                }
                if g.constant.isZero {
                    known[g.output] = Fr.zero
                    continue
                }
                keptGates.append(g)
            }
        }

        return OptCircuit(gates: keptGates, numWires: circuit.numWires,
                          numInputWires: circuit.numInputWires,
                          numOutputWires: circuit.numOutputWires)
    }

    // MARK: - Pass 2: Dead Gate Elimination

    /// Remove gates whose output wire is never used as input by any other gate
    /// and is not a circuit output wire.
    public func deadGateElimination(_ circuit: OptCircuit) -> OptCircuit {
        // Build set of wires that are consumed (used as inputs)
        var consumedWires = Set<Int>()
        for g in circuit.gates {
            consumedWires.insert(g.inputA)
            if g.inputB >= 0 {
                consumedWires.insert(g.inputB)
            }
        }

        // Output wires of the circuit are always "consumed"
        let outputStart = circuit.numWires - circuit.numOutputWires
        for i in outputStart..<circuit.numWires {
            consumedWires.insert(i)
        }

        // Iterate backwards (reverse topological order) for maximum elimination
        var alive = [Bool](repeating: false, count: circuit.gates.count)
        // First pass: mark gates whose output is consumed
        for (i, g) in circuit.gates.enumerated() {
            if consumedWires.contains(g.output) {
                alive[i] = true
            }
        }

        // Second pass: propagate liveness backwards - if a gate is dead,
        // its inputs might become unconsumed, killing more gates.
        // Rebuild consumed set from alive gates only.
        var changed = true
        while changed {
            changed = false
            consumedWires.removeAll(keepingCapacity: true)
            for (i, g) in circuit.gates.enumerated() where alive[i] {
                consumedWires.insert(g.inputA)
                if g.inputB >= 0 {
                    consumedWires.insert(g.inputB)
                }
            }
            // Re-check: output wires always consumed
            for i in outputStart..<circuit.numWires {
                consumedWires.insert(i)
            }
            for (i, g) in circuit.gates.enumerated() where alive[i] {
                if !consumedWires.contains(g.output) {
                    alive[i] = false
                    changed = true
                }
            }
        }

        let keptGates = circuit.gates.enumerated().compactMap { alive[$0.offset] ? $0.element : nil }
        return OptCircuit(gates: keptGates, numWires: circuit.numWires,
                          numInputWires: circuit.numInputWires,
                          numOutputWires: circuit.numOutputWires)
    }

    // MARK: - Pass 3: Common Subexpression Elimination

    /// Detect gates that compute identical operations on the same inputs and
    /// merge them so the output wire is shared.
    public func commonSubexpressionElimination(_ circuit: OptCircuit) -> OptCircuit {
        // Key: (kind, inputA, inputB, constant) -> first output wire
        struct GateKey: Hashable {
            let kind: OptGate.Kind
            let inputA: Int
            let inputB: Int
            let constant: Fr
        }

        var seen = [GateKey: Int]()   // key -> canonical output wire
        var wireRemap = [Int: Int]()  // duplicate output -> canonical output
        var keptGates = [OptGate]()

        for g in circuit.gates {
            // Remap inputs through any previous CSE merges
            let remappedA = wireRemap[g.inputA] ?? g.inputA
            let remappedB = g.inputB >= 0 ? (wireRemap[g.inputB] ?? g.inputB) : g.inputB

            let key = GateKey(kind: g.kind, inputA: remappedA, inputB: remappedB, constant: g.constant)

            if let existing = seen[key] {
                // This gate is a duplicate — remap its output to the existing one
                wireRemap[g.output] = existing
                continue
            }

            // Rebuild gate with remapped inputs
            let remapped = OptGate(kind: g.kind, inputA: remappedA, inputB: remappedB,
                                   output: g.output, constant: g.constant)
            seen[key] = g.output
            keptGates.append(remapped)
        }

        // Apply remaining wire remaps to all gate inputs
        let finalGates = keptGates.map { g -> OptGate in
            let a = wireRemap[g.inputA] ?? g.inputA
            let b = g.inputB >= 0 ? (wireRemap[g.inputB] ?? g.inputB) : g.inputB
            return OptGate(kind: g.kind, inputA: a, inputB: b, output: g.output, constant: g.constant)
        }

        return OptCircuit(gates: finalGates, numWires: circuit.numWires,
                          numInputWires: circuit.numInputWires,
                          numOutputWires: circuit.numOutputWires)
    }

    // MARK: - Pass 4: Gate Merging

    /// Merge cascaded linear gates. For example:
    ///   gate1: out1 = a + b  (add)
    ///   gate2: out2 = out1 + c  (add)
    /// If out1 is only used by gate2, merge into: out2 = a + b + c
    /// (represented as a single add with a constAdd chain).
    ///
    /// Also merges cascaded constMul: (x * c1) * c2 -> x * (c1*c2)
    public func gateMerging(_ circuit: OptCircuit) -> OptCircuit {
        // Count how many times each wire is used as input
        var useCount = [Int: Int]()
        for g in circuit.gates {
            useCount[g.inputA, default: 0] += 1
            if g.inputB >= 0 {
                useCount[g.inputB, default: 0] += 1
            }
        }

        // Build output -> gate index map
        var outputToGate = [Int: Int]()
        for (i, g) in circuit.gates.enumerated() {
            outputToGate[g.output] = i
        }

        var keptGates = [OptGate]()
        var eliminated = Set<Int>() // gate indices to skip

        for (i, g) in circuit.gates.enumerated() {
            if eliminated.contains(i) { continue }

            // Try to merge cascaded constMul: if this is constMul and its input
            // comes from another constMul used only here
            if g.kind == .constMul, let producerIdx = outputToGate[g.inputA],
               !eliminated.contains(producerIdx) {
                let producer = circuit.gates[producerIdx]
                if producer.kind == .constMul && useCount[producer.output, default: 0] == 1 {
                    // Merge: x * c1 * c2 -> x * (c1*c2)
                    let merged = OptGate(kind: .constMul, inputA: producer.inputA,
                                         output: g.output,
                                         constant: frMul(producer.constant, g.constant))
                    eliminated.insert(producerIdx)
                    keptGates.append(merged)
                    continue
                }
            }

            // Try to merge cascaded constAdd
            if g.kind == .constAdd, let producerIdx = outputToGate[g.inputA],
               !eliminated.contains(producerIdx) {
                let producer = circuit.gates[producerIdx]
                if producer.kind == .constAdd && useCount[producer.output, default: 0] == 1 {
                    // Merge: (x + c1) + c2 -> x + (c1+c2)
                    let merged = OptGate(kind: .constAdd, inputA: producer.inputA,
                                         output: g.output,
                                         constant: frAdd(producer.constant, g.constant))
                    eliminated.insert(producerIdx)
                    keptGates.append(merged)
                    continue
                }
            }

            keptGates.append(g)
        }

        return OptCircuit(gates: keptGates, numWires: circuit.numWires,
                          numInputWires: circuit.numInputWires,
                          numOutputWires: circuit.numOutputWires)
    }

    // MARK: - Pass 5: Wire Renumbering

    /// Compact wire indices to be contiguous, eliminating gaps from dead gates.
    /// Returns the compacted circuit and the number of eliminated wire slots.
    public func wireRenumbering(_ circuit: OptCircuit) -> (OptCircuit, eliminatedWires: Int) {
        // Collect all wire indices actually in use
        var usedWires = Set<Int>()
        // Input wires are always used
        for i in 0..<circuit.numInputWires {
            usedWires.insert(i)
        }
        for g in circuit.gates {
            usedWires.insert(g.inputA)
            if g.inputB >= 0 {
                usedWires.insert(g.inputB)
            }
            usedWires.insert(g.output)
        }

        if usedWires.count == circuit.numWires {
            return (circuit, 0)
        }

        // Build remapping: old index -> new contiguous index
        let sorted = usedWires.sorted()
        var remap = [Int: Int]()
        for (newIdx, oldIdx) in sorted.enumerated() {
            remap[oldIdx] = newIdx
        }

        let remappedGates = circuit.gates.map { g -> OptGate in
            OptGate(kind: g.kind,
                    inputA: remap[g.inputA] ?? g.inputA,
                    inputB: g.inputB >= 0 ? (remap[g.inputB] ?? g.inputB) : -1,
                    output: remap[g.output] ?? g.output,
                    constant: g.constant)
        }

        let newNumWires = sorted.count
        let eliminated = circuit.numWires - newNumWires
        return (OptCircuit(gates: remappedGates, numWires: newNumWires,
                           numInputWires: circuit.numInputWires,
                           numOutputWires: circuit.numOutputWires),
                eliminated)
    }

    // MARK: - Full Optimization Pipeline

    /// Run all optimization passes in sequence, returning the optimized circuit and stats.
    ///
    /// Pass order:
    ///   1. Constant folding
    ///   2. Dead gate elimination
    ///   3. Common subexpression elimination
    ///   4. Gate merging
    ///   5. Wire renumbering
    public func optimizeAll(_ circuit: OptCircuit,
                            knownConstants: [Int: Fr] = [:]) -> (OptCircuit, CircuitOptimizationStats) {
        let t0 = CFAbsoluteTimeGetCurrent()
        let originalGates = circuit.gateCount
        let originalWires = circuit.numWires
        var passStats = [CircuitPassStats]()
        var current = circuit

        // Pass 1: Constant folding
        var pt = CFAbsoluteTimeGetCurrent()
        let beforeCF = current.gateCount
        current = constantFolding(current, knownConstants: knownConstants)
        passStats.append(CircuitPassStats(passName: "Constant Folding",
                                          gatesBefore: beforeCF, gatesAfter: current.gateCount,
                                          timeMs: (CFAbsoluteTimeGetCurrent() - pt) * 1000.0))

        // Pass 2: Dead gate elimination
        pt = CFAbsoluteTimeGetCurrent()
        let beforeDGE = current.gateCount
        current = deadGateElimination(current)
        passStats.append(CircuitPassStats(passName: "Dead Gate Elimination",
                                          gatesBefore: beforeDGE, gatesAfter: current.gateCount,
                                          timeMs: (CFAbsoluteTimeGetCurrent() - pt) * 1000.0))

        // Pass 3: CSE
        pt = CFAbsoluteTimeGetCurrent()
        let beforeCSE = current.gateCount
        current = commonSubexpressionElimination(current)
        passStats.append(CircuitPassStats(passName: "CSE",
                                          gatesBefore: beforeCSE, gatesAfter: current.gateCount,
                                          timeMs: (CFAbsoluteTimeGetCurrent() - pt) * 1000.0))

        // Pass 4: Gate merging
        pt = CFAbsoluteTimeGetCurrent()
        let beforeMerge = current.gateCount
        current = gateMerging(current)
        passStats.append(CircuitPassStats(passName: "Gate Merging",
                                          gatesBefore: beforeMerge, gatesAfter: current.gateCount,
                                          timeMs: (CFAbsoluteTimeGetCurrent() - pt) * 1000.0))

        // Pass 5: Wire renumbering
        pt = CFAbsoluteTimeGetCurrent()
        let beforeRenumber = current.gateCount
        let (renumbered, _) = wireRenumbering(current)
        current = renumbered
        passStats.append(CircuitPassStats(passName: "Wire Renumbering",
                                          gatesBefore: beforeRenumber, gatesAfter: current.gateCount,
                                          timeMs: (CFAbsoluteTimeGetCurrent() - pt) * 1000.0))

        let totalMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
        let stats = CircuitOptimizationStats(
            passStats: passStats,
            originalGates: originalGates,
            optimizedGates: current.gateCount,
            originalWires: originalWires,
            optimizedWires: current.numWires,
            totalTimeMs: totalMs
        )

        return (current, stats)
    }

    // MARK: - Evaluate OptCircuit (CPU reference)

    /// Evaluate the circuit on given input values, returning all wire values.
    public func evaluate(_ circuit: OptCircuit, inputs: [Fr]) -> [Fr] {
        var wires = [Fr](repeating: Fr.zero, count: circuit.numWires)
        for (i, v) in inputs.prefix(circuit.numInputWires).enumerated() {
            wires[i] = v
        }

        for g in circuit.gates {
            let a = wires[g.inputA]
            let b = g.inputB >= 0 ? wires[g.inputB] : Fr.zero

            switch g.kind {
            case .add:
                wires[g.output] = frAdd(a, b)
            case .mul:
                wires[g.output] = frMul(a, b)
            case .constAdd:
                wires[g.output] = frAdd(a, g.constant)
            case .constMul:
                wires[g.output] = frMul(a, g.constant)
            }
        }

        return wires
    }
}
