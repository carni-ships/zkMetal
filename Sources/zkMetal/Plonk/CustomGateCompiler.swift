// CustomGateCompiler -- Compiles high-level custom gate descriptions into PlonkCircuit
//
// Takes a circuit described using CustomGateTemplate instances and:
//   a. Expands each custom gate into the minimal set of Plonk arithmetic constraints
//   b. Assigns wire indices, managing auxiliary variables automatically
//   c. Generates the selector polynomials
//   d. Outputs a PlonkCircuit ready for proving
//
// Usage:
//   let compiler = CustomGateCompiler()
//   let x = compiler.addInput()
//   let y = compiler.addInput()
//   compiler.addCustomGate(BoolGate(), inputs: [x])
//   compiler.addCustomGate(RangeGate(bits: 8), inputs: [y], auxiliaryWitness: bitDecomposition)
//   let circuit = compiler.compile()

import Foundation
import NeonFieldOps

// MARK: - Gate Entry

/// Internal representation of a custom gate added to the compiler.
struct CustomGateEntry {
    let template: any CustomGateTemplate
    /// User-provided input variable indices
    let inputVars: [Int]
    /// Auxiliary variable indices (auto-allocated)
    let auxVars: [Int]
    /// All variable indices in the order expected by the template
    let allVars: [Int]
}

// MARK: - CustomGateCompiler

/// Compiles circuits built from CustomGateTemplate instances into PlonkCircuit.
///
/// The compiler manages variable allocation, expands custom gates into arithmetic
/// constraints, and generates selector polynomial evaluations.
public class CustomGateCompiler {

    /// All registered gate entries
    private var entries: [CustomGateEntry] = []

    /// Explicit copy constraints (variable index pairs)
    private var copyConstraints: [(Int, Int)] = []

    /// Next variable index
    private var nextVar: Int = 0

    /// Witness values for all variables (set by the user or computed for aux vars)
    private var witnessValues: [Int: Fr] = [:]

    /// Public input variable indices
    private var publicInputs: [Int] = []

    /// Lookup tables registered during compilation
    private var lookupTables: [PlonkLookupTable] = []

    public init() {}

    // MARK: - Variable allocation

    /// Allocate a new input variable. Returns its index.
    public func addInput() -> Int {
        let v = nextVar
        nextVar += 1
        return v
    }

    /// Allocate multiple input variables.
    public func addInputs(_ count: Int) -> [Int] {
        let start = nextVar
        nextVar += count
        return Array(start..<(start + count))
    }

    /// Allocate a new variable with a known witness value.
    public func addVariable(value: Fr) -> Int {
        let v = nextVar
        nextVar += 1
        witnessValues[v] = value
        return v
    }

    /// Set the witness value for a variable.
    public func setWitness(_ variable: Int, value: Fr) {
        witnessValues[variable] = value
    }

    /// Mark a variable as a public input.
    public func addPublicInput(_ variable: Int) {
        publicInputs.append(variable)
    }

    /// Add a copy constraint: two variables must hold equal values.
    public func assertEqual(_ a: Int, _ b: Int) {
        copyConstraints.append((a, b))
    }

    /// Register a lookup table. Returns the table ID.
    @discardableResult
    public func addLookupTable(values: [Fr]) -> Int {
        let id = lookupTables.count
        lookupTables.append(PlonkLookupTable(id: id, values: values))
        return id
    }

    // MARK: - Gate addition

    /// Add a custom gate to the circuit.
    ///
    /// The compiler automatically allocates auxiliary variables needed by the gate
    /// beyond the user-provided inputs.
    ///
    /// - Parameters:
    ///   - template: The custom gate template
    ///   - inputs: Input variable indices (provided by the user)
    ///   - auxiliaryWitness: Optional witness values for auxiliary variables.
    ///     If provided, must contain exactly (template.wireCount - inputs.count) values.
    /// - Returns: All variable indices used by the gate (inputs + allocated auxiliaries)
    @discardableResult
    public func addCustomGate(_ template: any CustomGateTemplate,
                               inputs: [Int],
                               auxiliaryWitness: [Fr] = []) -> [Int] {
        let auxCount = template.wireCount - inputs.count
        precondition(auxCount >= 0,
                     "\(template.name): provided \(inputs.count) inputs but gate needs \(template.wireCount)")

        // Allocate auxiliary variables
        var auxVars = [Int]()
        for i in 0..<auxCount {
            let v = nextVar
            nextVar += 1
            auxVars.append(v)
            if i < auxiliaryWitness.count {
                witnessValues[v] = auxiliaryWitness[i]
            }
        }

        let allVars = inputs + auxVars
        precondition(allVars.count >= template.wireCount)

        let entry = CustomGateEntry(
            template: template,
            inputVars: inputs,
            auxVars: auxVars,
            allVars: allVars
        )
        entries.append(entry)
        return allVars
    }

    // MARK: - Convenience methods

    /// Add a boolean constraint on a variable.
    @discardableResult
    public func addBoolConstraint(_ variable: Int) -> [Int] {
        addCustomGate(BoolGate(), inputs: [variable])
    }

    /// Add a range check on a variable (n-bit range).
    /// Requires bit decomposition witness values.
    @discardableResult
    public func addRangeCheck(_ variable: Int, bits: Int, bitValues: [Fr]) -> [Int] {
        let gate = RangeGate(bits: bits)
        // Compute accumulator witness values
        var accValues = [Fr]()
        if bits > 1 {
            var acc = bitValues[0]
            for i in 1..<bits {
                let coeff = frFromInt(1 << UInt64(i))
                acc = frAdd(acc, frMul(bitValues[i], coeff))
                accValues.append(acc)
            }
        }
        return addCustomGate(gate, inputs: [variable], auxiliaryWitness: bitValues + accValues)
    }

    /// Add a conditional select: out = sel ? a : b.
    /// Returns the output variable index.
    public func addConditionalSelect(selector: Int, a: Int, b: Int,
                                      selectorValue: Fr, aValue: Fr, bValue: Fr) -> Int {
        let gate = ConditionalSelectGateTemplate()
        let outValue: Fr
        if frEqual(selectorValue, Fr.one) {
            outValue = aValue
        } else {
            outValue = bValue
        }
        let diff = frSub(aValue, bValue)
        let prod = frMul(selectorValue, diff)

        let out = addVariable(value: outValue)
        let allVars = addCustomGate(gate, inputs: [selector, a, b, out],
                                     auxiliaryWitness: [diff, prod])
        return out
    }

    /// Add an XOR gate. Returns the output variable index.
    public func addXor(a: Int, b: Int, aValue: Fr, bValue: Fr) -> Int {
        let gate = XorGate()
        let two = frAdd(Fr.one, Fr.one)
        let outValue = frSub(frAdd(aValue, bValue), frMul(two, frMul(aValue, bValue)))
        let out = addVariable(value: outValue)
        let abValue = frMul(aValue, bValue)
        _ = addCustomGate(gate, inputs: [a, b, out], auxiliaryWitness: [abValue])
        return out
    }

    // MARK: - Compile

    /// Compile all custom gates into a PlonkCircuit.
    ///
    /// Expands each gate template into arithmetic constraints, assigns wire indices,
    /// and generates the final circuit.
    ///
    /// - Returns: A PlonkCircuit ready for proving.
    public func compile() -> PlonkCircuit {
        var allGates = [PlonkGate]()
        var allWireAssignments = [[Int]]()

        // Expand each custom gate entry
        for entry in entries {
            let constraints = entry.template.buildConstraints(vars: entry.allVars)
            for (gate, wires) in constraints {
                allGates.append(gate)
                // Ensure exactly 3 wires per gate
                var paddedWires = wires
                while paddedWires.count < 3 {
                    // Pad with the last wire (dummy)
                    paddedWires.append(paddedWires.last ?? 0)
                }
                allWireAssignments.append(Array(paddedWires.prefix(3)))
            }
        }

        // Handle empty circuit
        if allGates.isEmpty {
            let dummy = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero,
                                   qM: Fr.zero, qC: Fr.zero)
            allGates.append(dummy)
            allWireAssignments.append([0, 0, 0])
        }

        return PlonkCircuit(
            gates: allGates,
            copyConstraints: copyConstraints,
            wireAssignments: allWireAssignments,
            lookupTables: lookupTables,
            publicInputIndices: publicInputs
        )
    }

    /// Compile and pad to the next power of 2.
    public func compileAndPad() -> PlonkCircuit {
        return compile().padded()
    }

    // MARK: - Witness generation

    /// Get the full witness vector (variable index -> value).
    /// Missing values default to Fr.zero.
    public func getWitness() -> [Fr] {
        var witness = [Fr](repeating: Fr.zero, count: nextVar)
        for (idx, val) in witnessValues {
            if idx < witness.count {
                witness[idx] = val
            }
        }
        return witness
    }

    /// Verify all constraints are satisfied by the current witness.
    ///
    /// - Returns: true if all gate evaluations return zero.
    public func verify() -> Bool {
        let witness = getWitness()

        for entry in entries {
            // Build witness slice for this gate
            var gateWitness = [Fr]()
            for v in entry.allVars {
                gateWitness.append(v < witness.count ? witness[v] : Fr.zero)
            }

            let result = entry.template.evaluate(witness: gateWitness)
            if !frEqual(result, Fr.zero) {
                return false
            }
        }

        // Verify copy constraints
        for (a, b) in copyConstraints {
            let va = a < witness.count ? witness[a] : Fr.zero
            let vb = b < witness.count ? witness[b] : Fr.zero
            if !frEqual(va, vb) {
                return false
            }
        }

        return true
    }

    // MARK: - Statistics

    /// Total number of Plonk arithmetic constraints after expansion.
    public var totalConstraintCount: Int {
        var count = 0
        for entry in entries {
            let constraints = entry.template.buildConstraints(vars: entry.allVars)
            count += constraints.count
        }
        return count
    }

    /// Total number of variables allocated.
    public var totalVariableCount: Int { nextVar }

    /// Number of custom gates added.
    public var gateCount: Int { entries.count }

    /// Generate selector polynomial evaluations.
    ///
    /// Returns arrays of selector values (qL, qR, qO, qM, qC) for each gate row,
    /// suitable for polynomial interpolation.
    public func generateSelectorPolynomials() -> (qL: [Fr], qR: [Fr], qO: [Fr], qM: [Fr], qC: [Fr]) {
        var qLvals = [Fr]()
        var qRvals = [Fr]()
        var qOvals = [Fr]()
        var qMvals = [Fr]()
        var qCvals = [Fr]()

        for entry in entries {
            let constraints = entry.template.buildConstraints(vars: entry.allVars)
            for (gate, _) in constraints {
                qLvals.append(gate.qL)
                qRvals.append(gate.qR)
                qOvals.append(gate.qO)
                qMvals.append(gate.qM)
                qCvals.append(gate.qC)
            }
        }

        return (qL: qLvals, qR: qRvals, qO: qOvals, qM: qMvals, qC: qCvals)
    }
}
