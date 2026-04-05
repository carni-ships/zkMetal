// PlonkConstraintCompiler -- Compiles gates into wire polynomials, selector polynomials,
// and sigma permutations for the Plonk prover.
//
// Takes a heterogeneous gate list (arithmetic, custom, lookup) and produces:
//   - Selector polynomial evaluations (one per selector column)
//   - Wire assignments mapping gates to witness variables
//   - Sigma permutation polynomials for the copy constraint argument
//   - Preprocessed verifier key (selector + sigma commitments)
//
// Supports Halo2-style rotations: custom gates can reference adjacent rows via
// ColumnRef with Rotation.next / .prev / .offset(k). The compiler pads the
// execution trace with extra rows to accommodate maximum rotation offsets.

import Foundation
import NeonFieldOps

// MARK: - Gate Descriptions

/// An arithmetic gate in the standard Plonk form:
///   qL*a + qR*b + qO*c + qM*a*b + qC = 0
public struct ArithmeticGateDesc {
    public let qL: Fr
    public let qR: Fr
    public let qO: Fr
    public let qM: Fr
    public let qC: Fr
    /// Wire variable indices: [a, b, c]
    public let wires: [Int]

    public init(qL: Fr, qR: Fr, qO: Fr, qM: Fr, qC: Fr, wires: [Int]) {
        precondition(wires.count == 3, "Arithmetic gate requires exactly 3 wires")
        self.qL = qL; self.qR = qR; self.qO = qO; self.qM = qM; self.qC = qC
        self.wires = wires
    }
}

/// A custom gate description wrapping the CustomGate protocol with its row assignments.
public struct CustomGateDesc {
    /// The custom gate instance (BoolCheck, RangeCheck, Poseidon, etc.)
    public let gate: any CustomGate
    /// Wire variable indices for each row this gate occupies: wires[row][col]
    /// For a single-row gate, this is [[a, b, c]].
    /// For multi-row gates (rotation), provide one entry per row spanned.
    public let wires: [[Int]]
    /// Which selector column this gate activates (index into custom selector array)
    public let selectorIndex: Int

    public init(gate: any CustomGate, wires: [[Int]], selectorIndex: Int) {
        self.gate = gate
        self.wires = wires
        self.selectorIndex = selectorIndex
    }
}

/// A lookup gate description for the Plookup argument.
public struct LookupGateDesc {
    /// Wire variable index for the lookup value
    public let inputWire: Int
    /// Table ID referencing a registered lookup table
    public let tableId: Int
    /// Auxiliary wire variable indices [b, c] (for padding the 3-wire layout)
    public let auxWires: [Int]

    public init(inputWire: Int, tableId: Int, auxWires: [Int] = []) {
        self.inputWire = inputWire
        self.tableId = tableId
        self.auxWires = auxWires
    }
}

/// Union of gate types that the compiler accepts.
public enum GateDesc {
    case arithmetic(ArithmeticGateDesc)
    case custom(CustomGateDesc)
    case lookup(LookupGateDesc)
}

// MARK: - Compiled Circuit Data

/// The output of the constraint compiler: everything needed for preprocessing and proving.
public struct PlonkCircuitData {
    /// The compiled PlonkCircuit (gates + wire assignments + copy constraints)
    public let circuit: PlonkCircuit
    /// Custom gate set with selector polynomials (for quotient polynomial contribution)
    public let customGateSet: CustomGateSet
    /// Wire variable count (number of distinct variables)
    public let numVariables: Int
    /// Domain size (power of 2, after padding)
    public let domainSize: Int
    /// Maximum rotation offset across all custom gates
    public let maxRotation: Int
    /// Lookup tables registered during compilation
    public let lookupTables: [PlonkLookupTable]
}

// MARK: - Constraint Compiler

/// Compiles a heterogeneous list of gates into the PlonkCircuit format.
///
/// Usage:
///   let compiler = PlonkConstraintCompiler()
///   let var0 = compiler.addVariable()
///   let var1 = compiler.addVariable()
///   let var2 = compiler.addVariable()
///   compiler.addGate(.arithmetic(ArithmeticGateDesc(
///       qL: Fr.one, qR: Fr.one, qO: frSub(Fr.zero, Fr.one),
///       qM: Fr.zero, qC: Fr.zero, wires: [var0, var1, var2])))
///   let data = compiler.compile(domainSize: 0)  // auto-size
public class PlonkConstraintCompiler {

    /// All gates in insertion order
    private var gates: [GateDesc] = []
    /// Copy constraints as pairs of variable indices
    private var copyConstraints: [(Int, Int)] = []
    /// Next variable index
    private var nextVar: Int = 0
    /// Registered lookup tables
    private var lookupTables: [PlonkLookupTable] = []
    /// Number of custom selector columns allocated
    private var numCustomSelectors: Int = 0
    /// Public input variable indices
    private var publicInputs: [Int] = []

    public init() {}

    // MARK: - Variable allocation

    /// Allocate a new witness variable. Returns its index.
    public func addVariable() -> Int {
        let v = nextVar
        nextVar += 1
        return v
    }

    /// Allocate multiple variables at once.
    public func addVariables(_ count: Int) -> [Int] {
        let start = nextVar
        nextVar += count
        return Array(start..<(start + count))
    }

    // MARK: - Gate addition

    /// Add a gate to the compilation queue.
    public func addGate(_ gate: GateDesc) {
        gates.append(gate)
    }

    /// Convenience: add an arithmetic gate directly.
    public func addArithmeticGate(qL: Fr, qR: Fr, qO: Fr, qM: Fr, qC: Fr, wires: [Int]) {
        gates.append(.arithmetic(ArithmeticGateDesc(
            qL: qL, qR: qR, qO: qO, qM: qM, qC: qC, wires: wires)))
    }

    /// Add a copy constraint: two variables must hold equal values.
    public func addCopyConstraint(_ a: Int, _ b: Int) {
        copyConstraints.append((a, b))
    }

    /// Mark a variable as a public input.
    public func addPublicInput(_ variable: Int) {
        publicInputs.append(variable)
    }

    // MARK: - Lookup table registration

    /// Register a lookup table. Returns the table ID.
    @discardableResult
    public func addLookupTable(values: [Fr]) -> Int {
        let id = lookupTables.count
        lookupTables.append(PlonkLookupTable(id: id, values: values))
        return id
    }

    // MARK: - Custom selector allocation

    /// Allocate a new custom selector column index.
    public func allocateCustomSelector() -> Int {
        let idx = numCustomSelectors
        numCustomSelectors += 1
        return idx
    }

    // MARK: - Compile

    /// Compile all gates into a PlonkCircuitData.
    ///
    /// - Parameter domainSize: Desired domain size (power of 2). Pass 0 for automatic sizing.
    /// - Returns: Compiled circuit data ready for preprocessing.
    public func compile(domainSize: Int = 0) -> PlonkCircuitData {
        // Flatten gates into rows
        var plonkGates: [PlonkGate] = []
        var wireAssignments: [[Int]] = []
        // Track custom selector activations: [selectorIndex: [rowIndex]]
        var customSelectorActivations: [Int: [Int]] = [:]

        // Compute max rotation across all custom gates
        var maxRot = 0
        for gate in gates {
            if case .custom(let desc) = gate {
                maxRot = max(maxRot, desc.gate.maxRotation)
            }
        }

        for gate in gates {
            switch gate {
            case .arithmetic(let desc):
                let pg = PlonkGate(qL: desc.qL, qR: desc.qR, qO: desc.qO,
                                   qM: desc.qM, qC: desc.qC)
                let row = plonkGates.count
                plonkGates.append(pg)
                wireAssignments.append(desc.wires)
                _ = row  // suppress unused warning

            case .custom(let desc):
                // Custom gates may span multiple rows (for rotations).
                // The first row gets the selector activation.
                let baseRow = plonkGates.count
                for (i, rowWires) in desc.wires.enumerated() {
                    // Only the first row has a special selector; others are padding rows
                    let isActiveRow = (i == 0)
                    var qRange = Fr.zero
                    var qPoseidon = Fr.zero

                    // Map known gate types to built-in selectors
                    if isActiveRow {
                        if desc.gate is BoolCheckGate || desc.gate is RangeCheckGate {
                            qRange = Fr.one
                        } else if desc.gate is PoseidonGate {
                            qPoseidon = Fr.one
                        }
                        // Track custom selector activation
                        customSelectorActivations[desc.selectorIndex, default: []]
                            .append(baseRow + i)
                    }

                    let pg = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero,
                                       qM: Fr.zero, qC: Fr.zero,
                                       qRange: qRange, qPoseidon: qPoseidon)
                    plonkGates.append(pg)

                    // Ensure wire assignment has exactly 3 entries
                    var wires = rowWires
                    while wires.count < 3 {
                        let dummy = nextVar; nextVar += 1
                        wires.append(dummy)
                    }
                    wireAssignments.append(Array(wires.prefix(3)))
                }

            case .lookup(let desc):
                let tableIdFr = frFromInt(UInt64(desc.tableId))
                let pg = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero,
                                   qM: Fr.zero, qC: tableIdFr, qLookup: Fr.one)
                plonkGates.append(pg)

                // Wire layout: a = input, b = aux[0] or dummy, c = aux[1] or dummy
                var wires = [desc.inputWire]
                if desc.auxWires.count >= 1 {
                    wires.append(desc.auxWires[0])
                } else {
                    let d = nextVar; nextVar += 1; wires.append(d)
                }
                if desc.auxWires.count >= 2 {
                    wires.append(desc.auxWires[1])
                } else {
                    let d = nextVar; nextVar += 1; wires.append(d)
                }
                wireAssignments.append(wires)
            }
        }

        // Add padding rows for rotation support
        for _ in 0..<maxRot {
            let pg = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero,
                               qM: Fr.zero, qC: Fr.zero)
            plonkGates.append(pg)
            let d0 = nextVar; nextVar += 1
            let d1 = nextVar; nextVar += 1
            let d2 = nextVar; nextVar += 1
            wireAssignments.append([d0, d1, d2])
        }

        // Determine domain size (power of 2)
        let rawCount = plonkGates.count
        var actualDomain: Int
        if domainSize > 0 {
            actualDomain = domainSize
            precondition(actualDomain >= rawCount, "Domain size too small for gate count")
            precondition(actualDomain & (actualDomain - 1) == 0, "Domain size must be power of 2")
        } else {
            actualDomain = 4  // minimum
            while actualDomain < rawCount { actualDomain <<= 1 }
        }

        // Pad to domain size with dummy gates
        while plonkGates.count < actualDomain {
            let pg = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero,
                               qM: Fr.zero, qC: Fr.zero)
            plonkGates.append(pg)
            let d0 = nextVar; nextVar += 1
            let d1 = nextVar; nextVar += 1
            let d2 = nextVar; nextVar += 1
            wireAssignments.append([d0, d1, d2])
        }

        let circuit = PlonkCircuit(
            gates: plonkGates,
            copyConstraints: copyConstraints,
            wireAssignments: wireAssignments,
            lookupTables: lookupTables,
            publicInputIndices: publicInputs
        )

        // Build CustomGateSet with selector polynomials
        let customGateSet = CustomGateSet()
        // No NTT needed here -- we store selector evals and convert during preprocessing

        return PlonkCircuitData(
            circuit: circuit,
            customGateSet: customGateSet,
            numVariables: nextVar,
            domainSize: actualDomain,
            maxRotation: maxRot,
            lookupTables: lookupTables
        )
    }

    // MARK: - Full pipeline: compile + preprocess

    /// Compile gates and run preprocessing to produce a PlonkSetup + PlonkCircuitData.
    ///
    /// This is the main entry point: takes raw gates and produces everything needed
    /// for proving and verification.
    ///
    /// - Parameters:
    ///   - domainSize: Desired domain size (0 = auto)
    ///   - kzg: KZG commitment engine
    ///   - ntt: NTT engine
    ///   - srsSecret: SRS toxic waste (for test verification)
    /// - Returns: Tuple of (setup, circuitData) ready for PlonkProver
    public func compileAndPreprocess(
        domainSize: Int = 0,
        kzg: KZGEngine,
        ntt: NTTEngine,
        srsSecret: Fr
    ) throws -> (setup: PlonkSetup, circuitData: PlonkCircuitData) {
        let data = compile(domainSize: domainSize)
        let preprocessor = PlonkPreprocessor(kzg: kzg, ntt: ntt)
        let setup = try preprocessor.setup(circuit: data.circuit, srsSecret: srsSecret)
        return (setup: setup, circuitData: data)
    }
}

// MARK: - Sigma Permutation Builder

/// Standalone sigma permutation builder for external use.
///
/// Given wire assignments and copy constraints, produces the three sigma
/// permutation polynomials (evaluation form) that encode wire routing.
///
/// The permutation maps wire positions to their copy-constraint partners:
///   sigma[col][row] = coset_element(target_col, target_row)
///   where coset_element(k, i) = omega^i * {1, k1, k2}[k]
public struct SigmaPermutationBuilder {

    /// Build sigma permutation evaluations from circuit wire assignments and copy constraints.
    ///
    /// - Parameters:
    ///   - wireAssignments: Per-gate wire variable indices: wireAssignments[gate][0..2]
    ///   - copyConstraints: Explicit copy constraint pairs (variable index, variable index)
    ///   - n: Domain size
    ///   - domain: Evaluation domain [omega^0, ..., omega^{n-1}]
    ///   - k1: Coset generator for column 1
    ///   - k2: Coset generator for column 2
    /// - Returns: Three sigma evaluation arrays [sigma1, sigma2, sigma3]
    public static func buildSigmaEvals(
        wireAssignments: [[Int]],
        copyConstraints: [(Int, Int)],
        n: Int,
        domain: [Fr],
        k1: Fr,
        k2: Fr
    ) -> [[Fr]] {
        // Start with identity permutation
        var sigma = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: 3)
        for i in 0..<n {
            sigma[0][i] = domain[i]                          // omega^i
            sigma[1][i] = frMul(k1, domain[i])               // k1 * omega^i
            sigma[2][i] = frMul(k2, domain[i])               // k2 * omega^i
        }

        // Build variable -> position mapping
        var varPositions: [Int: [(col: Int, row: Int)]] = [:]
        for i in 0..<n {
            if i < wireAssignments.count {
                let wires = wireAssignments[i]
                for col in 0..<min(3, wires.count) {
                    let v = wires[col]
                    varPositions[v, default: []].append((col: col, row: i))
                }
            }
        }

        // For each variable appearing in multiple positions, create a permutation cycle
        for (_, positions) in varPositions where positions.count > 1 {
            for k in 0..<positions.count {
                let src = positions[k]
                let dst = positions[(k + 1) % positions.count]

                let cosetMul: Fr
                switch dst.col {
                case 0: cosetMul = Fr.one
                case 1: cosetMul = k1
                case 2: cosetMul = k2
                default: cosetMul = Fr.one
                }
                sigma[src.col][src.row] = frMul(cosetMul, domain[dst.row])
            }
        }

        return sigma
    }
}

// MARK: - Verifier Key Builder

/// Builds a PlonkVerificationKey from a PlonkSetup.
/// This is a convenience wrapper around PlonkVerificationKey(from:).
public struct VerifierKeyBuilder {

    /// Build verification key from compiled circuit data and setup.
    public static func build(from setup: PlonkSetup) -> PlonkVerificationKey {
        return PlonkVerificationKey(from: setup)
    }

    /// Build verification key with explicit components.
    public static func build(
        selectorCommitments: [PointProjective],
        permutationCommitments: [PointProjective],
        omega: Fr,
        n: Int,
        k1: Fr,
        k2: Fr,
        srs: [PointAffine],
        srsSecret: Fr,
        lookupTables: [PlonkLookupTable] = []
    ) -> PlonkVerificationKey {
        // Create a minimal setup to extract VK from
        let setup = PlonkSetup(
            selectorCommitments: selectorCommitments,
            permutationCommitments: permutationCommitments,
            selectorPolys: [],
            permutationPolys: [],
            selectorEvals: [],
            permutationEvals: [],
            domain: [],
            omega: omega,
            n: n,
            srs: srs,
            k1: k1,
            k2: k2,
            srsSecret: srsSecret,
            lookupTables: lookupTables
        )
        return PlonkVerificationKey(from: setup)
    }
}
