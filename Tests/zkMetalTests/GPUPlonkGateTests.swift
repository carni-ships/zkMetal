// GPUPlonkGateTests — Tests for GPU-accelerated Plonk gate evaluation engine
//
// Tests cover:
//   1. Arithmetic gate satisfaction (qL*a + qR*b + qO*c + qM*a*b + qC = 0)
//   2. Range gate (boolean constraint)
//   3. Failing gate detection
//   4. Multi-gate circuit with mixed gate types
//   5. Selector isolation validation

import zkMetal
import Foundation

public func runGPUPlonkGateTests() {
    suite("GPU Plonk Gate Engine")

    let engine = GPUPlonkGateEngine()

    // ========== Test 1: Arithmetic gate satisfaction ==========
    // Gate: 1*a + 1*b + (-1)*c + 0*a*b + 0 = 0  =>  a + b - c = 0
    // Witness: a=3, b=5, c=8  =>  3 + 5 - 8 = 0
    do {
        let negOne = frSub(Fr.zero, Fr.one)
        let gate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)
        let circuit = PlonkCircuit(
            gates: [gate],
            copyConstraints: [],
            wireAssignments: [[0, 1, 2]]
        )
        let witness = [frFromInt(3), frFromInt(5), frFromInt(8)]

        let result = engine.evaluateArithmeticGates(circuit: circuit, witness: witness)
        expect(result.isSatisfied, "Arithmetic gate: a + b - c = 0 with (3,5,8)")
        expect(result.failingRows.isEmpty, "Arithmetic gate: no failing rows")
    }

    // ========== Test 2: Arithmetic multiplication gate ==========
    // Gate: 0*a + 0*b + (-1)*c + 1*a*b + 0 = 0  =>  a*b - c = 0
    // Witness: a=4, b=7, c=28
    do {
        let negOne = frSub(Fr.zero, Fr.one)
        let gate = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: negOne, qM: Fr.one, qC: Fr.zero)
        let circuit = PlonkCircuit(
            gates: [gate],
            copyConstraints: [],
            wireAssignments: [[0, 1, 2]]
        )
        let witness = [frFromInt(4), frFromInt(7), frFromInt(28)]

        let result = engine.evaluateArithmeticGates(circuit: circuit, witness: witness)
        expect(result.isSatisfied, "Arithmetic gate: a*b - c = 0 with (4,7,28)")
    }

    // ========== Test 3: Range gate (boolean check) ==========
    // Wire a must be 0 or 1 when qRange is active.
    do {
        // Valid: a=0
        let gate0 = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero,
                               qRange: Fr.one)
        // Valid: a=1
        let gate1 = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero,
                               qRange: Fr.one)
        let circuit = PlonkCircuit(
            gates: [gate0, gate1],
            copyConstraints: [],
            wireAssignments: [[0, 1, 2], [3, 4, 5]]
        )
        let witness = [Fr.zero, Fr.zero, Fr.zero, Fr.one, Fr.zero, Fr.zero]

        let result = engine.evaluateRangeGates(circuit: circuit, witness: witness)
        expect(result.isSatisfied, "Range gate: a=0 and a=1 both pass boolean check")
    }

    // ========== Test 4: Failing gate detection ==========
    // a + b - c = 0, but we provide c=99 (wrong)
    do {
        let negOne = frSub(Fr.zero, Fr.one)
        let gate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)
        let circuit = PlonkCircuit(
            gates: [gate],
            copyConstraints: [],
            wireAssignments: [[0, 1, 2]]
        )
        let witness = [frFromInt(3), frFromInt(5), frFromInt(99)]

        let result = engine.evaluateArithmeticGates(circuit: circuit, witness: witness)
        expect(!result.isSatisfied, "Failing gate: a + b != c detected")
        expect(result.failingRows.count == 1, "Failing gate: exactly 1 failing row")
        expect(result.failingRows[0] == 0, "Failing gate: row 0 fails")
    }

    // ========== Test 5: Failing range gate ==========
    // a=2 is not boolean
    do {
        let gate = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero,
                              qRange: Fr.one)
        let circuit = PlonkCircuit(
            gates: [gate],
            copyConstraints: [],
            wireAssignments: [[0, 1, 2]]
        )
        let witness = [frFromInt(2), Fr.zero, Fr.zero]

        let result = engine.evaluateRangeGates(circuit: circuit, witness: witness)
        expect(!result.isSatisfied, "Range gate: a=2 fails boolean check")
        expect(result.failingRows.count == 1, "Range gate: 1 failing row for a=2")
    }

    // ========== Test 6: Multi-gate circuit ==========
    // Gate 0: a + b = c  (addition)
    // Gate 1: a * b = c  (multiplication)
    // Gate 2: constant gate: 0 + 0 + 0 + 0 + 0 = 0
    do {
        let negOne = frSub(Fr.zero, Fr.one)
        let addGate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)
        let mulGate = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: negOne, qM: Fr.one, qC: Fr.zero)
        let zeroGate = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero)

        let circuit = PlonkCircuit(
            gates: [addGate, mulGate, zeroGate],
            copyConstraints: [],
            wireAssignments: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        )
        // Gate 0: 3+5=8, Gate 1: 4*7=28, Gate 2: 0+0+0=0
        let witness = [frFromInt(3), frFromInt(5), frFromInt(8),
                       frFromInt(4), frFromInt(7), frFromInt(28),
                       Fr.zero, Fr.zero, Fr.zero]

        let result = engine.evaluateArithmeticGates(circuit: circuit, witness: witness)
        expect(result.isSatisfied, "Multi-gate: all 3 gates satisfied")
        expect(result.residuals.count == 3, "Multi-gate: 3 residuals")
    }

    // ========== Test 7: Selector isolation ==========
    // Valid: only one special selector active per row
    do {
        let g0 = PlonkGate(qL: Fr.one, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero,
                            qRange: Fr.one)
        let g1 = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero,
                            qPoseidon: Fr.one)
        let g2 = PlonkGate(qL: Fr.one, qR: Fr.one, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero)

        let circuit = PlonkCircuit(
            gates: [g0, g1, g2],
            copyConstraints: [],
            wireAssignments: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        )

        let violations = engine.checkSelectorIsolation(circuit: circuit)
        expect(violations.isEmpty, "Selector isolation: no violations with separate selectors")
    }

    // ========== Test 8: Selector isolation violation ==========
    // Invalid: qRange AND qPoseidon both active on same row
    do {
        let badGate = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero,
                                 qRange: Fr.one, qPoseidon: Fr.one)
        let circuit = PlonkCircuit(
            gates: [badGate],
            copyConstraints: [],
            wireAssignments: [[0, 1, 2]]
        )

        let violations = engine.checkSelectorIsolation(circuit: circuit)
        expect(violations.count == 1, "Selector isolation: detects dual-selector violation")
        expect(violations[0] == 0, "Selector isolation: violation at row 0")
    }

    // ========== Test 9: Poseidon gate ==========
    // c = a^5 when qPoseidon is active
    do {
        let a = frFromInt(3)
        let a2 = frSqr(a)
        let a4 = frSqr(a2)
        let a5 = frMul(a, a4) // 3^5 = 243

        let gate = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero,
                              qPoseidon: Fr.one)
        let circuit = PlonkCircuit(
            gates: [gate],
            copyConstraints: [],
            wireAssignments: [[0, 1, 2]]
        )
        let witness = [a, Fr.zero, a5]

        let result = engine.evaluatePoseidonGates(circuit: circuit, witness: witness)
        expect(result.isSatisfied, "Poseidon gate: c = a^5 satisfied for a=3")
    }

    // ========== Test 10: Poseidon gate failure ==========
    do {
        let gate = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero,
                              qPoseidon: Fr.one)
        let circuit = PlonkCircuit(
            gates: [gate],
            copyConstraints: [],
            wireAssignments: [[0, 1, 2]]
        )
        // a=3, c=42 (wrong, should be 243)
        let witness = [frFromInt(3), Fr.zero, frFromInt(42)]

        let result = engine.evaluatePoseidonGates(circuit: circuit, witness: witness)
        expect(!result.isSatisfied, "Poseidon gate: wrong c detected")
    }

    // ========== Test 11: Combined gate evaluation ==========
    do {
        let negOne = frSub(Fr.zero, Fr.one)

        // Row 0: arithmetic gate a+b-c=0 with a=3, b=5, c=8
        let arithGate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)
        // Row 1: range gate with a=1 (boolean)
        let rangeGate = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero,
                                   qRange: Fr.one)
        // Row 2: poseidon gate with a=2, c=32 (2^5)
        let a2 = frFromInt(2)
        let c2 = frFromInt(32)
        let posGate = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero,
                                 qPoseidon: Fr.one)

        let circuit = PlonkCircuit(
            gates: [arithGate, rangeGate, posGate],
            copyConstraints: [],
            wireAssignments: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        )
        let witness = [frFromInt(3), frFromInt(5), frFromInt(8),
                       Fr.one, Fr.zero, Fr.zero,
                       a2, Fr.zero, c2]

        let result = engine.evaluateAllGates(circuit: circuit, witness: witness)
        expect(result.isSatisfied, "Combined evaluation: all gate types pass")
    }

    // ========== Test 12: Wire permutation check ==========
    do {
        let negOne = frSub(Fr.zero, Fr.one)
        let gate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)

        // Two gates: gate0.c == gate1.a (both should be 8)
        // Copy constraint: position (0*3+2=2) == position (1*3+0=3)
        let circuit = PlonkCircuit(
            gates: [gate, gate],
            copyConstraints: [(2, 3)],
            wireAssignments: [[0, 1, 2], [2, 3, 4]]
        )
        // var2 = 8, shared between gate0.c and gate1.a
        let witness = [frFromInt(3), frFromInt(5), frFromInt(8), frFromInt(7), frFromInt(15)]

        let violations = engine.checkWirePermutation(circuit: circuit, witness: witness)
        expect(violations.isEmpty, "Wire permutation: consistent copy constraint passes")
    }

    // ========== Test 13: Wire permutation violation ==========
    do {
        let negOne = frSub(Fr.zero, Fr.one)
        let gate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)

        // Copy constraint says gate0.c == gate1.a, but they have different variables
        let circuit = PlonkCircuit(
            gates: [gate, gate],
            copyConstraints: [(2, 3)],
            wireAssignments: [[0, 1, 2], [3, 4, 5]]
        )
        // var2=8, var3=99 -- different values!
        let witness = [frFromInt(3), frFromInt(5), frFromInt(8), frFromInt(99), frFromInt(7), frFromInt(106)]

        let violations = engine.checkWirePermutation(circuit: circuit, witness: witness)
        expect(violations.count == 1, "Wire permutation: violation detected for mismatched values")
    }

    // ========== Test 14: Single gate evaluateArithmeticGate ==========
    do {
        // Constant gate: qC = 0 => always satisfies with all-zero wires
        let gate = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero)
        let r = engine.evaluateArithmeticGate(gate: gate, a: frFromInt(42), b: frFromInt(99), c: frFromInt(7))
        expect(frEqual(r, Fr.zero), "Single gate eval: zero selectors always zero")

        // Non-zero constant: qC=5 => residual = 5
        let gate2 = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: frFromInt(5))
        let r2 = engine.evaluateArithmeticGate(gate: gate2, a: Fr.zero, b: Fr.zero, c: Fr.zero)
        expect(frEqual(r2, frFromInt(5)), "Single gate eval: qC=5 gives residual 5")
    }

    // ========== Test 15: evaluateByType dispatch ==========
    do {
        let negOne = frSub(Fr.zero, Fr.one)
        let gate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)
        let circuit = PlonkCircuit(
            gates: [gate],
            copyConstraints: [],
            wireAssignments: [[0, 1, 2]]
        )
        let witness = [frFromInt(3), frFromInt(5), frFromInt(8)]

        let arithResult = engine.evaluateByType(.arithmetic, circuit: circuit, witness: witness)
        expect(arithResult.isSatisfied, "evaluateByType(.arithmetic): passes")

        let rangeResult = engine.evaluateByType(.range, circuit: circuit, witness: witness)
        expect(rangeResult.isSatisfied, "evaluateByType(.range): passes (no range selector)")

        let ecResult = engine.evaluateByType(.ellipticCurve, circuit: circuit, witness: witness)
        expect(ecResult.isSatisfied, "evaluateByType(.ellipticCurve): passes (trivially)")
    }
}
