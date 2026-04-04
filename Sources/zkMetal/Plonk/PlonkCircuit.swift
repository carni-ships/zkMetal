// PlonkCircuit — Circuit representation for Plonk-style proofs
//
// Each gate has selectors (qL, qR, qO, qM, qC) defining the constraint:
//   qL*a + qR*b + qO*c + qM*a*b + qC = 0
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

    public init(qL: Fr, qR: Fr, qO: Fr, qM: Fr, qC: Fr) {
        self.qL = qL; self.qR = qR; self.qO = qO; self.qM = qM; self.qC = qC
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

    public var numGates: Int { gates.count }

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

        return PlonkCircuit(gates: paddedGates, copyConstraints: copyConstraints, wireAssignments: paddedWires)
    }
}

// MARK: - Circuit Builder

public class PlonkCircuitBuilder {
    public var gates: [PlonkGate] = []
    public var copyConstraints: [(Int, Int)] = []
    public var wireAssignments: [[Int]] = []  // [gateIdx] -> [aVar, bVar, cVar]
    public var nextVariable: Int = 0

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

    /// Build the circuit (does NOT pad; call .padded() on result if needed)
    public func build() -> PlonkCircuit {
        return PlonkCircuit(
            gates: gates,
            copyConstraints: copyConstraints,
            wireAssignments: wireAssignments
        )
    }
}
