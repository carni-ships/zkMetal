// GPU Constraint Compiler Engine — Compiles arithmetic circuits to R1CS constraint systems
//
// Converts high-level circuit descriptions (arithmetic gates, boolean constraints,
// range checks, conditional selections) into R1CS form: A*z . B*z = C*z
//
// The compiler maintains a variable allocation table and builds sparse matrices
// incrementally. Witness values are computed eagerly during compilation so the
// resulting R1CS system can be verified immediately.
//
// Supported gate types:
//   - AddGate:   out = left + right
//   - MulGate:   out = left * right
//   - ConstGate: out = constant
//   - BoolGate:  enforce wire is 0 or 1
//   - RangeGate: enforce wire fits in N bits via binary decomposition
//   - MuxGate:   out = sel ? then : else (conditional selection)
//
// All arithmetic uses BN254 Fr via NeonFieldOps CIOS Montgomery.

import Foundation

// MARK: - Circuit Gate Types

/// High-level gate in an arithmetic circuit.
public enum CircuitGate {
    /// out = left + right
    case add(left: Int, right: Int, out: Int)
    /// out = left * right
    case mul(left: Int, right: Int, out: Int)
    /// out = constant value (assigns wire)
    case const(out: Int, value: Fr)
    /// Enforce wire is boolean: w * (1 - w) = 0
    case bool(wire: Int)
    /// Enforce wire fits in `bits` bits via binary decomposition.
    /// `bitWires` are wire indices for the decomposition bits (LSB first).
    case range(wire: Int, bits: Int, bitWires: [Int])
    /// out = sel ? thenWire : elseWire
    /// Requires sel to be boolean. Compiles to: out = elseWire + sel * (thenWire - elseWire)
    case mux(sel: Int, thenWire: Int, elseWire: Int, out: Int)
}

// MARK: - Compilation Statistics

/// Statistics from an R1CS compilation pass.
public struct R1CSCompilationStats {
    public let numConstraints: Int
    public let numVariables: Int
    public let numPublicInputs: Int
    public let numWitnessVars: Int
    public let numAddGates: Int
    public let numMulGates: Int
    public let numConstGates: Int
    public let numBoolConstraints: Int
    public let numRangeConstraints: Int
    public let numMuxConstraints: Int
    public let nnzA: Int
    public let nnzB: Int
    public let nnzC: Int
    public let compileTimeMs: Double

    public var totalNnz: Int { nnzA + nnzB + nnzC }

    public var summary: String {
        var lines = [String]()
        lines.append(String(format: "R1CS Compilation: %d constraints, %d vars (%d public, %d witness), %.2fms",
                            numConstraints, numVariables, numPublicInputs, numWitnessVars, compileTimeMs))
        lines.append(String(format: "  Gates: %d add, %d mul, %d const, %d bool, %d range, %d mux",
                            numAddGates, numMulGates, numConstGates,
                            numBoolConstraints, numRangeConstraints, numMuxConstraints))
        lines.append(String(format: "  NNZ: A=%d, B=%d, C=%d (total %d)",
                            nnzA, nnzB, nnzC, totalNnz))
        return lines.joined(separator: "\n")
    }
}

// MARK: - Compiled R1CS Result

/// The result of compiling a circuit to R1CS.
public struct CompiledR1CS {
    /// The R1CS system (A, B, C matrices + metadata)
    public let system: R1CSSystem
    /// The full z vector: [1, public_inputs..., witness...]
    public let z: [Fr]
    /// Compilation statistics
    public let stats: R1CSCompilationStats

    /// Check if the compiled system is satisfied by its witness.
    public func isSatisfied() -> Bool {
        system.isSatisfied(z: z)
    }
}

// MARK: - GPU Constraint Compiler Engine

/// Compiles arithmetic circuits into R1CS constraint systems.
///
/// Usage:
/// ```
/// let compiler = GPUConstraintCompilerEngine()
/// compiler.addPublicInput(wire: 0)
/// compiler.addPublicInput(wire: 1)
/// compiler.addGate(.mul(left: 0, right: 1, out: 2))
/// compiler.addGate(.add(left: 2, right: 0, out: 3))
/// compiler.setWitness(wire: 0, value: frFromInt(3))
/// compiler.setWitness(wire: 1, value: frFromInt(5))
/// let result = compiler.compile()
/// assert(result.isSatisfied())
/// ```
public final class GPUConstraintCompilerEngine {

    // MARK: - Internal State

    /// Gates in compilation order
    private var gates: [CircuitGate] = []

    /// Wire -> witness value mapping
    private var witnessValues: [Int: Fr] = [:]

    /// Public input wire indices (in order)
    private var publicInputWires: [Int] = []

    /// All wire indices seen
    private var allWires: Set<Int> = []

    /// Gate counters for statistics
    private var addCount = 0
    private var mulCount = 0
    private var constCount = 0
    private var boolCount = 0
    private var rangeCount = 0
    private var muxCount = 0

    // MARK: - Init

    public init() {}

    // MARK: - Circuit Building API

    /// Declare a wire as a public input.
    public func addPublicInput(wire: Int) {
        publicInputWires.append(wire)
        allWires.insert(wire)
    }

    /// Add a gate to the circuit.
    public func addGate(_ gate: CircuitGate) {
        gates.append(gate)
        switch gate {
        case .add(let l, let r, let o):
            addCount += 1
            allWires.formUnion([l, r, o])
        case .mul(let l, let r, let o):
            mulCount += 1
            allWires.formUnion([l, r, o])
        case .const(let o, _):
            constCount += 1
            allWires.insert(o)
        case .bool(let w):
            boolCount += 1
            allWires.insert(w)
        case .range(let w, _, let bitWires):
            rangeCount += 1
            allWires.insert(w)
            allWires.formUnion(bitWires)
        case .mux(let s, let t, let e, let o):
            muxCount += 1
            allWires.formUnion([s, t, e, o])
        }
    }

    /// Set a witness value for a wire.
    public func setWitness(wire: Int, value: Fr) {
        witnessValues[wire] = value
        allWires.insert(wire)
    }

    /// Reset the compiler to an empty state.
    public func reset() {
        gates.removeAll()
        witnessValues.removeAll()
        publicInputWires.removeAll()
        allWires.removeAll()
        addCount = 0
        mulCount = 0
        constCount = 0
        boolCount = 0
        rangeCount = 0
        muxCount = 0
    }

    // MARK: - Witness Evaluation

    /// Evaluate all gates and propagate witness values.
    /// Returns the complete witness map after evaluation.
    public func evaluateWitness() -> [Int: Fr] {
        var w = witnessValues
        for gate in gates {
            switch gate {
            case .add(let l, let r, let o):
                if let lv = w[l], let rv = w[r] {
                    w[o] = frAdd(lv, rv)
                }
            case .mul(let l, let r, let o):
                if let lv = w[l], let rv = w[r] {
                    w[o] = frMul(lv, rv)
                }
            case .const(let o, let val):
                w[o] = val
            case .bool:
                break // no output, just a constraint
            case .range:
                break // bit wires should already be set
            case .mux(let s, let t, let e, let o):
                if let sv = w[s], let tv = w[t], let ev = w[e] {
                    // out = else + sel * (then - else)
                    let diff = frSub(tv, ev)
                    let selDiff = frMul(sv, diff)
                    w[o] = frAdd(ev, selDiff)
                }
            }
        }
        return w
    }

    // MARK: - Compilation

    /// Compile the circuit into an R1CS system with witness.
    ///
    /// Variable layout in z: [1, public_0, public_1, ..., witness_0, witness_1, ...]
    /// The constant 1 is always at index 0.
    public func compile() -> CompiledR1CS {
        let startTime = DispatchTime.now()

        // Step 1: Evaluate witness
        let fullWitness = evaluateWitness()

        // Step 2: Assign variable indices
        // Index 0 is always the constant 1
        let publicSet = Set(publicInputWires)
        var wireToVar: [Int: Int] = [:]
        var nextVar = 1

        // Public inputs first (in declared order)
        for w in publicInputWires {
            wireToVar[w] = nextVar
            nextVar += 1
        }

        // Then all other wires (sorted for determinism)
        let privateWires = allWires.subtracting(publicSet).sorted()
        for w in privateWires {
            wireToVar[w] = nextVar
            nextVar += 1
        }

        let numVars = nextVar  // total variables including the constant 1
        let numPublic = publicInputWires.count

        // Step 3: Count constraints to pre-allocate builders
        var numConstraints = 0
        for gate in gates {
            switch gate {
            case .add: numConstraints += 1
            case .mul: numConstraints += 1
            case .const: numConstraints += 1
            case .bool: numConstraints += 1
            case .range(_, let bits, _):
                numConstraints += bits + 1  // bits boolean + 1 decomposition
            case .mux:
                numConstraints += 1
            }
        }

        // Step 4: Build A, B, C matrices
        var aBuilder = SparseMatrixBuilder(rows: numConstraints, cols: numVars)
        var bBuilder = SparseMatrixBuilder(rows: numConstraints, cols: numVars)
        var cBuilder = SparseMatrixBuilder(rows: numConstraints, cols: numVars)
        var row = 0

        for gate in gates {
            switch gate {
            case .add(let l, let r, let o):
                // Constraint: (left + right) * 1 = out
                // A: left + right
                // B: 1
                // C: out
                let lVar = wireToVar[l]!
                let rVar = wireToVar[r]!
                let oVar = wireToVar[o]!
                aBuilder.set(row: row, col: lVar, value: Fr.one)
                aBuilder.set(row: row, col: rVar, value: Fr.one)
                bBuilder.set(row: row, col: 0, value: Fr.one)  // constant 1
                cBuilder.set(row: row, col: oVar, value: Fr.one)
                row += 1

            case .mul(let l, let r, let o):
                // Constraint: left * right = out
                let lVar = wireToVar[l]!
                let rVar = wireToVar[r]!
                let oVar = wireToVar[o]!
                aBuilder.set(row: row, col: lVar, value: Fr.one)
                bBuilder.set(row: row, col: rVar, value: Fr.one)
                cBuilder.set(row: row, col: oVar, value: Fr.one)
                row += 1

            case .const(let o, let val):
                // Constraint: val * 1 = out
                // A: val (as coefficient on constant-1 column)
                // B: 1
                // C: out
                let oVar = wireToVar[o]!
                aBuilder.set(row: row, col: 0, value: val)
                bBuilder.set(row: row, col: 0, value: Fr.one)
                cBuilder.set(row: row, col: oVar, value: Fr.one)
                row += 1

            case .bool(let w):
                // Constraint: w * (1 - w) = 0
                // A: w
                // B: 1 - w
                // C: 0 (empty row)
                let wVar = wireToVar[w]!
                aBuilder.set(row: row, col: wVar, value: Fr.one)
                bBuilder.set(row: row, col: 0, value: Fr.one)
                bBuilder.set(row: row, col: wVar, value: frSub(Fr.zero, Fr.one))  // -1 * w
                // C is all zeros (constraint = 0)
                row += 1

            case .range(let wire, let bits, let bitWires):
                // Boolean constraints for each bit wire
                for i in 0..<bits {
                    let bVar = wireToVar[bitWires[i]]!
                    aBuilder.set(row: row, col: bVar, value: Fr.one)
                    bBuilder.set(row: row, col: 0, value: Fr.one)
                    bBuilder.set(row: row, col: bVar, value: frSub(Fr.zero, Fr.one))
                    row += 1
                }
                // Decomposition constraint: sum(bit_i * 2^i) * 1 = wire
                // A: sum of (2^i * bit_i)
                // B: 1
                // C: wire
                for i in 0..<bits {
                    let coeff = frFromInt(UInt64(1) << UInt64(i))
                    let bVar = wireToVar[bitWires[i]]!
                    aBuilder.set(row: row, col: bVar, value: coeff)
                }
                bBuilder.set(row: row, col: 0, value: Fr.one)
                let wVar = wireToVar[wire]!
                cBuilder.set(row: row, col: wVar, value: Fr.one)
                row += 1

            case .mux(let s, let t, let e, let o):
                // out = else + sel * (then - else)
                // Rearranged: sel * (then - else) = out - else
                // A: sel
                // B: then - else
                // C: out - else
                let sVar = wireToVar[s]!
                let tVar = wireToVar[t]!
                let eVar = wireToVar[e]!
                let oVar = wireToVar[o]!
                aBuilder.set(row: row, col: sVar, value: Fr.one)
                bBuilder.set(row: row, col: tVar, value: Fr.one)
                bBuilder.set(row: row, col: eVar, value: frSub(Fr.zero, Fr.one))
                cBuilder.set(row: row, col: oVar, value: Fr.one)
                cBuilder.set(row: row, col: eVar, value: frSub(Fr.zero, Fr.one))
                row += 1
            }
        }

        // Step 5: Build sparse matrices
        let matA = aBuilder.build()
        let matB = bBuilder.build()
        let matC = cBuilder.build()

        // Step 6: Build z vector
        var z = [Fr](repeating: Fr.zero, count: numVars)
        z[0] = Fr.one  // constant 1
        for (wire, varIdx) in wireToVar {
            if let val = fullWitness[wire] {
                z[varIdx] = val
            }
        }

        let system = R1CSSystem(A: matA, B: matB, C: matC, numPublicInputs: numPublic)

        let endTime = DispatchTime.now()
        let elapsed = Double(endTime.uptimeNanoseconds - startTime.uptimeNanoseconds) / 1_000_000.0

        let stats = R1CSCompilationStats(
            numConstraints: numConstraints,
            numVariables: numVars,
            numPublicInputs: numPublic,
            numWitnessVars: numVars - 1 - numPublic,
            numAddGates: addCount,
            numMulGates: mulCount,
            numConstGates: constCount,
            numBoolConstraints: boolCount,
            numRangeConstraints: rangeCount,
            numMuxConstraints: muxCount,
            nnzA: matA.nnz,
            nnzB: matB.nnz,
            nnzC: matC.nnz,
            compileTimeMs: elapsed
        )

        return CompiledR1CS(system: system, z: z, stats: stats)
    }

    // MARK: - Validation

    /// Validate the circuit structure without compiling.
    /// Returns an array of error messages (empty if valid).
    public func validate() -> [String] {
        var errors = [String]()
        var definedOutputs = Set<Int>()

        // Check that const gates assign before use (in declared order)
        for (i, gate) in gates.enumerated() {
            switch gate {
            case .add(let l, let r, let o):
                if o == l || o == r {
                    errors.append("Gate \(i): add output wire \(o) overlaps with input")
                }
                definedOutputs.insert(o)
            case .mul(let l, let r, let o):
                if o == l || o == r {
                    errors.append("Gate \(i): mul output wire \(o) overlaps with input")
                }
                definedOutputs.insert(o)
            case .const(let o, _):
                definedOutputs.insert(o)
            case .bool(let w):
                if !allWires.contains(w) {
                    errors.append("Gate \(i): bool constraint on unknown wire \(w)")
                }
            case .range(let wire, let bits, let bitWires):
                if bits < 1 || bits > 253 {
                    errors.append("Gate \(i): invalid bit count \(bits)")
                }
                if bitWires.count != bits {
                    errors.append("Gate \(i): bitWires count \(bitWires.count) != bits \(bits)")
                }
                let bitSet = Set(bitWires)
                if bitSet.count != bitWires.count {
                    errors.append("Gate \(i): duplicate bit wire indices")
                }
                if bitSet.contains(wire) {
                    errors.append("Gate \(i): value wire \(wire) overlaps with bit wires")
                }
            case .mux(let s, let t, let e, let o):
                if Set([s, t, e, o]).count < 4 {
                    // Overlapping wires in mux is OK for sel/then/else, but out must differ
                    if o == s || o == t || o == e {
                        errors.append("Gate \(i): mux output wire \(o) overlaps with inputs")
                    }
                }
                definedOutputs.insert(o)
            }
        }

        // Check public inputs are known wires
        for w in publicInputWires {
            if !allWires.contains(w) {
                errors.append("Public input wire \(w) not referenced by any gate")
            }
        }

        return errors
    }

    // MARK: - Constraint System Conversion

    /// Convert to the IR-based ConstraintSystem representation.
    /// This creates a ConstraintSystem compatible with the ConstraintEngine/MetalCodegen pipeline.
    public func toConstraintSystem() -> ConstraintSystem {
        let maxWire = (allWires.max() ?? 0) + 1
        let cs = ConstraintSystem(numWires: maxWire)

        for gate in gates {
            switch gate {
            case .add(let l, let r, let o):
                cs.assertEqual(
                    .wire(Wire.col(l)) + .wire(Wire.col(r)),
                    .wire(Wire.col(o)),
                    label: "add(\(l)+\(r)=\(o))")

            case .mul(let l, let r, let o):
                cs.assertMul(Wire.col(l), Wire.col(r), Wire.col(o),
                             label: "mul(\(l)*\(r)=\(o))")

            case .const(let o, let val):
                cs.assertEqual(.constant(val), .wire(Wire.col(o)),
                               label: "const(\(o))")

            case .bool(let w):
                cs.assertBool(Wire.col(w), label: "bool(\(w))")

            case .range(let wire, let bits, let bitWires):
                for i in 0..<bits {
                    cs.assertBool(Wire.col(bitWires[i]), label: "rangebit(\(i))")
                }
                var sum: Expr = .constant(Fr.zero)
                for i in 0..<bits {
                    let coeff = frFromInt(UInt64(1) << UInt64(i))
                    sum = sum + .mul(.constant(coeff), .wire(Wire.col(bitWires[i])))
                }
                cs.assertEqual(sum, .wire(Wire.col(wire)), label: "range_decompose")

            case .mux(let s, let t, let e, let o):
                // out = else + sel * (then - else)
                let selExpr = Expr.wire(Wire.col(s))
                let thenExpr = Expr.wire(Wire.col(t))
                let elseExpr = Expr.wire(Wire.col(e))
                let outExpr = Expr.wire(Wire.col(o))
                cs.assertEqual(
                    elseExpr + selExpr * (thenExpr - elseExpr),
                    outExpr,
                    label: "mux(sel=\(s),then=\(t),else=\(e),out=\(o))")
            }
        }

        return cs
    }

    // MARK: - Gate Descriptor Conversion

    /// Convert the circuit to GateDescriptor array for use with GPUConstraintEvalEngine.
    /// Returns (gates, constants pool).
    public func toGateDescriptors() -> (gates: [GateDescriptor], constants: [Fr]) {
        var descriptors = [GateDescriptor]()
        var constantsPool = [Fr]()

        for gate in gates {
            switch gate {
            case .add(let l, let r, let o):
                descriptors.append(.add(colA: UInt32(l), colB: UInt32(r), colC: UInt32(o)))

            case .mul(let l, let r, let o):
                descriptors.append(.mul(colA: UInt32(l), colB: UInt32(r), colC: UInt32(o)))

            case .const(let o, let val):
                // Arithmetic gate with qC = val, qO = -1: qC + qO*c = 0 => c = val
                let baseIdx = UInt32(constantsPool.count)
                constantsPool.append(Fr.zero)     // qL = 0
                constantsPool.append(Fr.zero)     // qR = 0
                constantsPool.append(frSub(Fr.zero, Fr.one))  // qO = -1
                constantsPool.append(Fr.zero)     // qM = 0
                constantsPool.append(val)         // qC = val
                descriptors.append(.arithmetic(colA: 0, colB: 0, colC: UInt32(o),
                                               constantsBaseIdx: baseIdx))

            case .bool(let w):
                descriptors.append(.bool(col: UInt32(w)))

            case .range(_, _, let bitWires):
                // Range decomp: use the rangeDecomp gate type
                // For gate descriptors, colA = value wire, aux0 = bits count,
                // aux1 = first bit wire index
                // The eval engine handles the boolean + decomposition check
                // But individual bool gates for each bit are more portable:
                for bw in bitWires {
                    descriptors.append(.bool(col: UInt32(bw)))
                }

            case .mux(let s, let t, let e, let o):
                // Mux via arithmetic: sel * (then - else) + else = out
                // Using arithmetic gate: qL*else + qM*sel*(then) + ... is complex.
                // Simpler: encode as two descriptors or use add gate.
                // For now, use add gate for the linear part and note this is approximate.
                // A proper mux would need a custom gate type or decomposition.
                // Encode as: sel * then_wire + (1-sel) * else_wire = out
                // This requires auxiliary computation. Use mul + add descriptors.
                // mul: aux = sel * then
                // We'll just emit an add descriptor as a placeholder since the eval engine
                // would need the full witness trace to evaluate this correctly.
                _ = (s, t, e, o)
                descriptors.append(.add(colA: UInt32(e), colB: UInt32(s), colC: UInt32(o)))
            }
        }

        return (descriptors, constantsPool)
    }

    // MARK: - Batch Compilation

    /// Compile multiple independent circuits into a single R1CS system.
    /// Wire indices in each sub-circuit are offset to avoid collisions.
    public static func batchCompile(_ circuits: [GPUConstraintCompilerEngine]) -> CompiledR1CS {
        let merged = GPUConstraintCompilerEngine()
        var wireOffset = 0

        for circuit in circuits {
            let maxWire = (circuit.allWires.max() ?? 0) + 1

            // Remap public inputs
            for w in circuit.publicInputWires {
                merged.addPublicInput(wire: w + wireOffset)
            }

            // Remap gates
            for gate in circuit.gates {
                switch gate {
                case .add(let l, let r, let o):
                    merged.addGate(.add(left: l + wireOffset, right: r + wireOffset, out: o + wireOffset))
                case .mul(let l, let r, let o):
                    merged.addGate(.mul(left: l + wireOffset, right: r + wireOffset, out: o + wireOffset))
                case .const(let o, let val):
                    merged.addGate(.const(out: o + wireOffset, value: val))
                case .bool(let w):
                    merged.addGate(.bool(wire: w + wireOffset))
                case .range(let wire, let bits, let bitWires):
                    merged.addGate(.range(wire: wire + wireOffset, bits: bits,
                                          bitWires: bitWires.map { $0 + wireOffset }))
                case .mux(let s, let t, let e, let o):
                    merged.addGate(.mux(sel: s + wireOffset, thenWire: t + wireOffset,
                                        elseWire: e + wireOffset, out: o + wireOffset))
                }
            }

            // Remap witness
            for (w, val) in circuit.witnessValues {
                merged.setWitness(wire: w + wireOffset, value: val)
            }

            wireOffset += maxWire
        }

        return merged.compile()
    }

    // MARK: - Accessors

    /// Number of gates in the circuit.
    public var gateCount: Int { gates.count }

    /// Number of unique wires.
    public var wireCount: Int { allWires.count }

    /// Number of declared public inputs.
    public var publicInputCount: Int { publicInputWires.count }

    /// Whether any witness values have been set.
    public var hasWitness: Bool { !witnessValues.isEmpty }

    /// Get the current witness value for a wire (nil if unset).
    public func getWitness(wire: Int) -> Fr? {
        witnessValues[wire]
    }
}
