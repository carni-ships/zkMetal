// PlonkCircuit — Circuit representation for Plonk-style proofs
//
// Each gate has selectors (qL, qR, qO, qM, qC) defining the constraint:
//   qL*a + qR*b + qO*c + qM*a*b + qC = 0
//
// Custom gate selectors extend the base constraint:
//   qRange: range gate — a*(1-a)=0 (boolean check), b reconstructs from limbs
//   qLookup: lookup gate — marks this gate as part of a lookup argument
//   qPoseidon: Poseidon S-box gate — c = a^5 (qPoseidon*(c - a^5) = 0)
//
// Copy constraints enforce wire equality across gates via permutation arguments.

import Foundation
import NeonFieldOps

// MARK: - Gate

public struct PlonkGate {
    public let qL: Fr   // left wire selector
    public let qR: Fr   // right wire selector
    public let qO: Fr   // output wire selector
    public let qM: Fr   // multiplication selector
    public let qC: Fr   // constant selector
    public let qRange: Fr    // range check selector
    public let qLookup: Fr   // lookup selector
    public let qPoseidon: Fr // Poseidon S-box selector

    public init(qL: Fr, qR: Fr, qO: Fr, qM: Fr, qC: Fr,
                qRange: Fr = Fr.zero, qLookup: Fr = Fr.zero, qPoseidon: Fr = Fr.zero) {
        self.qL = qL; self.qR = qR; self.qO = qO; self.qM = qM; self.qC = qC
        self.qRange = qRange; self.qLookup = qLookup; self.qPoseidon = qPoseidon
    }
}

// MARK: - Lookup Table

/// A lookup table for the Plookup-style argument.
/// Values in the table are field elements that wire values must belong to.
public struct PlonkLookupTable {
    public let id: Int
    public let values: [Fr]

    public init(id: Int, values: [Fr]) {
        self.id = id
        self.values = values
    }
}

// MARK: - Wire indexing

/// Each gate i has 3 wires: a (left), b (right), c (output).
/// Wire indices in the global permutation:
///   a-wires: 0..<n, b-wires: n..<2n, c-wires: 2n..<3n
/// where n = number of gates.

// MARK: - Circuit

public struct PlonkCircuit {
    public let gates: [PlonkGate]
    /// Copy constraints: pairs of (gateIndex * 3 + wireType) that must hold equal values.
    /// wireType: 0=a, 1=b, 2=c
    public let copyConstraints: [(Int, Int)]
    /// Wire assignments: wireAssignments[gateIdx][0..2] = variable index for a, b, c
    public let wireAssignments: [[Int]]
    /// Lookup tables used by lookup gates
    public let lookupTables: [PlonkLookupTable]

    public var numGates: Int { gates.count }

    public init(gates: [PlonkGate], copyConstraints: [(Int, Int)], wireAssignments: [[Int]],
                lookupTables: [PlonkLookupTable] = []) {
        self.gates = gates
        self.copyConstraints = copyConstraints
        self.wireAssignments = wireAssignments
        self.lookupTables = lookupTables
    }

    /// Pad circuit to next power of 2 (required for NTT-based polynomial ops)
    public func padded() -> PlonkCircuit {
        let n = gates.count
        var logN = 0
        while (1 << logN) < n { logN += 1 }
        let paddedN = 1 << logN
        if paddedN == n { return self }

        // Pad with dummy gates: 0*a + 0*b + 0*c + 0*a*b + 0 = 0
        var paddedGates = gates
        var paddedWires = wireAssignments
        let nextVar = (wireAssignments.flatMap { $0 }.max() ?? -1) + 1
        for i in 0..<(paddedN - n) {
            paddedGates.append(PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero))
            // Each dummy gate gets unique dummy wire variables
            let base = nextVar + i * 3
            paddedWires.append([base, base + 1, base + 2])
        }

        return PlonkCircuit(gates: paddedGates, copyConstraints: copyConstraints,
                            wireAssignments: paddedWires, lookupTables: lookupTables)
    }
}

// MARK: - Circuit Builder

public class PlonkCircuitBuilder {
    public var gates: [PlonkGate] = []
    public var copyConstraints: [(Int, Int)] = []
    public var wireAssignments: [[Int]] = []  // [gateIdx] -> [aVar, bVar, cVar]
    public var nextVariable: Int = 0
    public var lookupTables: [PlonkLookupTable] = []

    public init() {}

    /// Allocate a new input variable
    public func addInput() -> Int {
        let v = nextVariable
        nextVariable += 1
        return v
    }

    /// Add gate: a + b = c. Returns output variable.
    @discardableResult
    public func add(_ a: Int, _ b: Int) -> Int {
        let c = nextVariable; nextVariable += 1
        // qL=1, qR=1, qO=-1, qM=0, qC=0: a + b - c = 0
        let gate = PlonkGate(
            qL: Fr.one, qR: Fr.one,
            qO: frSub(Fr.zero, Fr.one),
            qM: Fr.zero, qC: Fr.zero
        )
        wireAssignments.append([a, b, c])
        gates.append(gate)
        return c
    }

    /// Mul gate: a * b = c. Returns output variable.
    @discardableResult
    public func mul(_ a: Int, _ b: Int) -> Int {
        let c = nextVariable; nextVariable += 1
        // qL=0, qR=0, qO=-1, qM=1, qC=0: a*b - c = 0
        let gate = PlonkGate(
            qL: Fr.zero, qR: Fr.zero,
            qO: frSub(Fr.zero, Fr.one),
            qM: Fr.one, qC: Fr.zero
        )
        wireAssignments.append([a, b, c])
        gates.append(gate)
        return c
    }

    /// Constant gate: output = value. Returns output variable.
    @discardableResult
    public func constant(_ value: Fr) -> Int {
        let c = nextVariable; nextVariable += 1
        // Use a dummy input variable for the a,b wires
        let dummy = nextVariable; nextVariable += 1
        // qL=0, qR=0, qO=-1, qM=0, qC=value: -c + value = 0 => c = value
        let gate = PlonkGate(
            qL: Fr.zero, qR: Fr.zero,
            qO: frSub(Fr.zero, Fr.one),
            qM: Fr.zero, qC: value
        )
        wireAssignments.append([dummy, dummy, c])
        gates.append(gate)
        return c
    }

    /// Copy constraint: variables a and b must have equal values.
    public func assertEqual(_ a: Int, _ b: Int) {
        copyConstraints.append((a, b))
    }

    // MARK: - Range Check Gate

    /// Range check: proves 0 <= value < 2^bits using custom range gates.
    ///
    /// Decomposes value into `bits` boolean limbs, adds boolean constraints
    /// (limb*(1-limb)=0) via range gates, and a reconstruction constraint
    /// (sum of limb_i * 2^i = value).
    ///
    /// Returns the variable holding the original value (for chaining).
    /// The witness must assign correct bit decomposition values to the
    /// auxiliary limb variables.
    @discardableResult
    public func rangeCheck(_ value: Int, bits: Int) -> (value: Int, limbVars: [Int]) {
        // Allocate boolean limb variables
        var limbVars = [Int]()
        for _ in 0..<bits {
            let limb = nextVariable; nextVariable += 1
            limbVars.append(limb)
        }

        // For each limb: add a boolean constraint gate
        // qRange=1 activates: a*(1-a) = 0 (a is the limb on wire a)
        // We use wire a = limb, wire b = unused (dummy), wire c = unused (dummy)
        for i in 0..<bits {
            let dummy = nextVariable; nextVariable += 1
            let gate = PlonkGate(
                qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero,
                qRange: Fr.one
            )
            wireAssignments.append([limbVars[i], dummy, dummy])
            gates.append(gate)
        }

        // Reconstruction constraint: sum(limb_i * 2^i) - value = 0
        // We build this with a chain of add gates:
        //   acc_0 = limb_0
        //   acc_i = acc_{i-1} + limb_i * 2^i
        // Final: assertEqual(acc_{bits-1}, value)
        //
        // Using standard gates: for each bit, add limb_i * 2^i
        // First, create weighted limb constants and accumulate
        var accVar = limbVars[0]  // limb_0 * 2^0 = limb_0

        for i in 1..<bits {
            // We need: acc_new = acc_old + limb_i * 2^i
            // Use gate: qL=1, qR=2^i, qO=-1: acc_old + 2^i * limb_i - acc_new = 0
            let acc_new = nextVariable; nextVariable += 1
            let coeff = frFromInt(1 << UInt64(i))
            let gate = PlonkGate(
                qL: Fr.one, qR: coeff,
                qO: frSub(Fr.zero, Fr.one),
                qM: Fr.zero, qC: Fr.zero
            )
            wireAssignments.append([accVar, limbVars[i], acc_new])
            gates.append(gate)
            accVar = acc_new
        }

        // Assert reconstruction equals original value
        assertEqual(accVar, value)

        return (value: value, limbVars: limbVars)
    }

    // MARK: - Lookup Gate

    /// Register a lookup table. Returns the table ID.
    @discardableResult
    public func addLookupTable(values: [Fr]) -> Int {
        let id = lookupTables.count
        lookupTables.append(PlonkLookupTable(id: id, values: values))
        return id
    }

    /// Lookup gate: proves that the value in variable `input` exists in the
    /// lookup table with the given tableId.
    ///
    /// The constraint is: qLookup * (prod_{t in table} (input - t)) = 0
    /// In practice, the prover provides an auxiliary witness that is the
    /// index into the table, and we verify input = table[index].
    ///
    /// For the simplified Plonk custom gate approach, we mark the gate
    /// with qLookup=1 and wire a = input. The lookup argument is enforced
    /// during the quotient polynomial computation.
    ///
    /// Returns the input variable for chaining.
    @discardableResult
    public func lookup(_ input: Int, tableId: Int) -> Int {
        let dummy = nextVariable; nextVariable += 1
        // Wire a = value to look up, wire b = tableId encoded as constant,
        // wire c = dummy
        let tableIdFr = frFromInt(UInt64(tableId))
        let gate = PlonkGate(
            qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero,
            qC: tableIdFr,
            qLookup: Fr.one
        )
        wireAssignments.append([input, dummy, dummy])
        gates.append(gate)
        return input
    }

    // MARK: - Poseidon S-box Gate

    /// Poseidon S-box gate: output = input^5.
    /// Custom constraint: qPoseidon * (c - a^5) = 0
    /// Wire a = input, wire c = output (= input^5), wire b = auxiliary (a^2).
    ///
    /// Returns the output variable containing input^5.
    @discardableResult
    public func poseidonSbox(_ input: Int) -> Int {
        let sq = nextVariable; nextVariable += 1      // a^2 (auxiliary)
        let output = nextVariable; nextVariable += 1  // a^5

        // qPoseidon=1 activates: c - a^5 = 0
        // We also constrain b = a^2 via qM=1, qR=-1: a*a - b = 0
        // But that's in a separate gate. For the custom S-box gate:
        //   wire a = input, wire b = a^2 (auxiliary), wire c = a^5 (output)
        //   constraint: qPoseidon * (c - a * b^2) = 0
        //   where b = a^2, so a * b^2 = a * a^4 = a^5
        //
        // We need an auxiliary constraint that b = a^2:
        //   Gate 1: a*a - b = 0  (standard mul gate)
        //   Gate 2: qPoseidon=1: c - a * b^2 = 0 (custom S-box gate)

        // Gate 1: b = a * a (standard mul)
        let mulGate = PlonkGate(
            qL: Fr.zero, qR: Fr.zero,
            qO: frSub(Fr.zero, Fr.one),
            qM: Fr.one, qC: Fr.zero
        )
        wireAssignments.append([input, input, sq])
        gates.append(mulGate)

        // Gate 2: Poseidon custom gate
        // Constraint: qPoseidon * (c - a * b * b) = 0
        // wire a = input, wire b = sq (=a^2), wire c = output (=a^5)
        let sboxGate = PlonkGate(
            qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero,
            qPoseidon: Fr.one
        )
        wireAssignments.append([input, sq, output])
        gates.append(sboxGate)

        return output
    }

    /// Poseidon2 external linear layer as gates.
    /// M_E * [a,b,c] = [2a+b+c, a+2b+c, a+b+2c]
    /// = [a + sum, b + sum, c + sum] where sum = a+b+c
    ///
    /// Takes 3 input variables, returns 3 output variables.
    public func poseidonExternalLinearLayer(_ inputs: [Int]) -> [Int] {
        precondition(inputs.count == 3)
        let a = inputs[0], b = inputs[1], c = inputs[2]

        // sum = a + b
        let ab = add(a, b)
        // sum = ab + c
        let sum = add(ab, c)

        // out0 = a + sum  (= 2a + b + c)
        let out0 = add(a, sum)
        // out1 = b + sum  (= a + 2b + c)
        let out1 = add(b, sum)
        // out2 = c + sum  (= a + b + 2c)
        let out2 = add(c, sum)

        return [out0, out1, out2]
    }

    /// Poseidon2 internal linear layer as gates.
    /// M_I * [a,b,c] = [2a+b+c, a+2b+c, a+b+3c]
    public func poseidonInternalLinearLayer(_ inputs: [Int]) -> [Int] {
        precondition(inputs.count == 3)
        let a = inputs[0], b = inputs[1], c = inputs[2]

        let ab = add(a, b)
        let sum = add(ab, c)

        let out0 = add(a, sum)     // 2a+b+c
        let out1 = add(b, sum)     // a+2b+c
        let out2 = add(c, add(sum, c))  // a+b+3c (= sum + 2c = (a+b+c) + 2c)

        return [out0, out1, out2]
    }

    /// Add round constant to a variable: out = in + rc.
    @discardableResult
    public func addConstant(_ input: Int, _ rc: Fr) -> Int {
        let out = nextVariable; nextVariable += 1
        // qL=1, qO=-1, qC=rc: a - c + rc = 0 => c = a + rc
        let gate = PlonkGate(
            qL: Fr.one, qR: Fr.zero,
            qO: frSub(Fr.zero, Fr.one),
            qM: Fr.zero, qC: rc
        )
        let dummy = nextVariable; nextVariable += 1
        wireAssignments.append([input, dummy, out])
        gates.append(gate)
        return out
    }

    /// Build the circuit (does NOT pad; call .padded() on result if needed)
    public func build() -> PlonkCircuit {
        return PlonkCircuit(
            gates: gates,
            copyConstraints: copyConstraints,
            wireAssignments: wireAssignments,
            lookupTables: lookupTables
        )
    }
}
