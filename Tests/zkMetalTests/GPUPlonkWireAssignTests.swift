// GPUPlonkWireAssignTests — Tests for GPU-accelerated Plonk wire assignment engine
//
// Tests cover:
//   1. Basic wire assignment from circuit + witness
//   2. UltraPlonk extended wire assignment (4+ wires)
//   3. Wire polynomial computation (eval -> coeff -> eval round-trip)
//   4. Wire rotation (shifted evaluations)
//   5. Inverse rotation
//   6. Public input injection
//   7. Public input polynomial construction
//   8. Gate constraint evaluation over assigned wires
//   9. Gate constraint with rotation (next-row access)
//  10. Polynomial evaluation at challenge point (Horner)
//  11. Vanishing polynomial evaluation
//  12. Lagrange basis evaluation
//  13. Batch NTT round-trip
//  14. Wire blinding
//  15. Coset evaluation
//  16. Quotient wire contribution
//  17. Wire commitment (small SRS)
//  18. Domain padding
//  19. Empty circuit edge case
//  20. Multi-gate stress test

import zkMetal
import Foundation

public func runGPUPlonkWireAssignTests() {
    suite("GPU Plonk Wire Assign Engine")

    testBasicWireAssignment()
    testWireAssignmentPadding()
    testUltraPlonkWireAssignment()
    testWireRotation()
    testInverseRotation()
    testRotationRoundTrip()
    testPublicInputInjection()
    testPublicInputPoly()
    testGateConstraintSatisfied()
    testGateConstraintViolation()
    testGateConstraintMulGate()
    testGateConstraintWithRotation()
    testPolynomialEvalAtPoint()
    testPolynomialEvalShifted()
    testVanishingPolyEval()
    testLagrangeBasisEval()
    testBatchNTTRoundTrip()
    testWireBlinding()
    testCosetEvaluation()
    testWireCommitment()
    testEmptyCircuit()
    testConfigDefaults()
    testMultiGateCircuit()
    testPublicInputInjectionBounds()
    testHornerEvalConstant()
    testWireAssignLargerCircuit()
    testQuotientWireContribution()
    testWirePolyComputation()
}

// MARK: - Basic Wire Assignment

private func testBasicWireAssignment() {
    let engine = GPUPlonkWireAssignEngine()

    // Circuit: 1 gate, a + b - c = 0
    // wireAssignments: gate 0 -> [var0, var1, var2]
    // witness: [3, 5, 8]
    let negOne = frSub(Fr.zero, Fr.one)
    let gate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)
    let circuit = PlonkCircuit(gates: [gate], copyConstraints: [],
                               wireAssignments: [[0, 1, 2]])
    let witness = [frFromInt(3), frFromInt(5), frFromInt(8)]

    let wireEvals = engine.assignWires(circuit: circuit, witness: witness)

    expect(wireEvals.count == 3, "3 wire columns for standard Plonk")
    // Domain padded to at least 4
    expect(wireEvals[0].count >= 4, "Domain padded to power of 2")
    // Wire 0 (a) at row 0 = 3
    expect(frEqual(wireEvals[0][0], frFromInt(3)), "Wire a[0] = 3")
    // Wire 1 (b) at row 0 = 5
    expect(frEqual(wireEvals[1][0], frFromInt(5)), "Wire b[0] = 5")
    // Wire 2 (c) at row 0 = 8
    expect(frEqual(wireEvals[2][0], frFromInt(8)), "Wire c[0] = 8")
    // Padded rows are zero
    expect(wireEvals[0][1].isZero, "Padded wire a[1] = 0")
}

// MARK: - Padding

private func testWireAssignmentPadding() {
    let engine = GPUPlonkWireAssignEngine()

    // 3 gates -> should pad to 4 (next power of 2)
    let gate = PlonkGate(qL: Fr.one, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero)
    let circuit = PlonkCircuit(
        gates: [gate, gate, gate],
        copyConstraints: [],
        wireAssignments: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    )
    let witness = (0..<9).map { frFromInt(UInt64($0 + 1)) }

    let wireEvals = engine.assignWires(circuit: circuit, witness: witness)

    expectEqual(wireEvals[0].count, 4, "3 gates padded to domain size 4")
    // Row 3 should be zero (padding)
    expect(wireEvals[0][3].isZero, "Padded row 3 wire a = 0")
    expect(wireEvals[1][3].isZero, "Padded row 3 wire b = 0")
    expect(wireEvals[2][3].isZero, "Padded row 3 wire c = 0")
}

// MARK: - UltraPlonk Extended Wires

private func testUltraPlonkWireAssignment() {
    let engine = GPUPlonkWireAssignEngine()

    // 4-wire assignment for UltraPlonk
    let wireAssignments = [[0, 1, 2, 3], [4, 5, 6, 7]]
    let witness = (0..<8).map { frFromInt(UInt64($0 + 10)) }

    let wireEvals = engine.assignUltraWires(
        wireAssignments: wireAssignments,
        witness: witness,
        numWires: 4,
        domainSize: 4
    )

    expectEqual(wireEvals.count, 4, "4 wire columns for UltraPlonk")
    expectEqual(wireEvals[0].count, 4, "Domain size 4")
    // Wire 0 row 0 = var0 = 10
    expect(frEqual(wireEvals[0][0], frFromInt(10)), "Ultra wire 0[0] = 10")
    // Wire 3 row 0 = var3 = 13
    expect(frEqual(wireEvals[3][0], frFromInt(13)), "Ultra wire 3[0] = 13")
    // Wire 0 row 1 = var4 = 14
    expect(frEqual(wireEvals[0][1], frFromInt(14)), "Ultra wire 0[1] = 14")
    // Padded rows (rows 2,3) are zero
    expect(wireEvals[0][2].isZero, "Ultra padded wire 0[2] = 0")
}

// MARK: - Wire Rotation

private func testWireRotation() {
    let engine = GPUPlonkWireAssignEngine()

    // evals[0] = [10, 20, 30, 40]
    let evals: [[Fr]] = [
        [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    ]

    let rotated = engine.computeRotations(evaluations: evals)

    expectEqual(rotated.count, 1, "1 wire column rotated")
    expectEqual(rotated[0].count, 4, "Same domain size after rotation")
    // rotated[0] = [20, 30, 40, 10] (shift left by 1, wrap around)
    expect(frEqual(rotated[0][0], frFromInt(20)), "Rotation: row 0 gets value from row 1")
    expect(frEqual(rotated[0][1], frFromInt(30)), "Rotation: row 1 gets value from row 2")
    expect(frEqual(rotated[0][2], frFromInt(40)), "Rotation: row 2 gets value from row 3")
    expect(frEqual(rotated[0][3], frFromInt(10)), "Rotation: row 3 wraps to row 0")
}

private func testInverseRotation() {
    let engine = GPUPlonkWireAssignEngine()

    let evals: [[Fr]] = [
        [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    ]

    let invRotated = engine.computeInverseRotations(evaluations: evals)

    // invRotated[0] = [40, 10, 20, 30] (shift right by 1)
    expect(frEqual(invRotated[0][0], frFromInt(40)), "Inv rotation: row 0 gets value from row n-1")
    expect(frEqual(invRotated[0][1], frFromInt(10)), "Inv rotation: row 1 gets value from row 0")
    expect(frEqual(invRotated[0][2], frFromInt(20)), "Inv rotation: row 2 gets value from row 1")
    expect(frEqual(invRotated[0][3], frFromInt(30)), "Inv rotation: row 3 gets value from row 2")
}

private func testRotationRoundTrip() {
    let engine = GPUPlonkWireAssignEngine()

    let evals: [[Fr]] = [
        [frFromInt(7), frFromInt(13), frFromInt(42), frFromInt(99)]
    ]

    // Rotate forward then inverse should recover original
    let rotated = engine.computeRotations(evaluations: evals)
    let recovered = engine.computeInverseRotations(evaluations: rotated)

    for i in 0..<4 {
        expect(frEqual(recovered[0][i], evals[0][i]),
               "Rotation round-trip recovers original at row \(i)")
    }
}

// MARK: - Public Input Injection

private func testPublicInputInjection() {
    let engine = GPUPlonkWireAssignEngine()

    // Wire evals: 3 wires, 4 rows, all initially zero
    var wireEvals = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: 4), count: 3)
    wireEvals[0][0] = frFromInt(99) // will be overwritten

    let config = WireAssignConfig(publicInputIndices: [0, 1])
    let publicInputs = [frFromInt(42), frFromInt(17)]

    let result = engine.injectPublicInputs(
        wireEvals: wireEvals,
        publicInputs: publicInputs,
        config: config
    )

    expect(frEqual(result.modifiedEvaluations[0][0], frFromInt(42)),
           "Public input 42 injected at row 0")
    expect(frEqual(result.modifiedEvaluations[0][1], frFromInt(17)),
           "Public input 17 injected at row 1")
    expectEqual(result.injectedRows.count, 2, "2 rows injected")
    expectEqual(result.injectedRows[0], 0, "First injection at row 0")
    expectEqual(result.injectedRows[1], 1, "Second injection at row 1")
    expectEqual(result.publicValues.count, 2, "2 public values")
    // Wire 1 and 2 should be unchanged
    expect(result.modifiedEvaluations[1][0].isZero, "Wire 1 unchanged by public input injection")
}

private func testPublicInputPoly() {
    let engine = GPUPlonkWireAssignEngine()

    let publicInputs = [frFromInt(5), frFromInt(11)]
    let publicIndices = [0, 2]
    let domainSize = 4

    let poly = engine.buildPublicInputPoly(
        publicInputs: publicInputs,
        publicInputIndices: publicIndices,
        domainSize: domainSize
    )

    expectEqual(poly.count, 4, "Public input poly has domain size 4")
    expect(frEqual(poly[0], frFromInt(5)), "PI poly[0] = 5")
    expect(poly[1].isZero, "PI poly[1] = 0")
    expect(frEqual(poly[2], frFromInt(11)), "PI poly[2] = 11")
    expect(poly[3].isZero, "PI poly[3] = 0")
}

private func testPublicInputInjectionBounds() {
    let engine = GPUPlonkWireAssignEngine()

    // Public input index out of bounds should be skipped
    let wireEvals = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: 4), count: 3)
    let config = WireAssignConfig(publicInputIndices: [0, 100]) // index 100 out of bounds
    let publicInputs = [frFromInt(42), frFromInt(17)]

    let result = engine.injectPublicInputs(
        wireEvals: wireEvals,
        publicInputs: publicInputs,
        config: config
    )

    expectEqual(result.injectedRows.count, 1, "Only 1 valid injection (index 100 skipped)")
    expect(frEqual(result.modifiedEvaluations[0][0], frFromInt(42)),
           "Valid injection at row 0 succeeded")
}

// MARK: - Gate Constraint Evaluation

private func testGateConstraintSatisfied() {
    let engine = GPUPlonkWireAssignEngine()

    // a + b - c = 0 with a=3, b=5, c=8
    let negOne = frSub(Fr.zero, Fr.one)
    let gate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)
    let circuit = PlonkCircuit(gates: [gate], copyConstraints: [],
                               wireAssignments: [[0, 1, 2]])
    let witness = [frFromInt(3), frFromInt(5), frFromInt(8)]

    let wireEvals = engine.assignWires(circuit: circuit, witness: witness)
    let result = engine.evaluateGateConstraints(circuit: circuit, wireEvals: wireEvals)

    expect(result.allSatisfied, "Gate a + b - c = 0 satisfied with (3, 5, 8)")
    expect(result.failingRows.isEmpty, "No failing rows")
}

private func testGateConstraintViolation() {
    let engine = GPUPlonkWireAssignEngine()

    // a + b - c = 0, but c=99 (wrong)
    let negOne = frSub(Fr.zero, Fr.one)
    let gate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)
    let circuit = PlonkCircuit(gates: [gate], copyConstraints: [],
                               wireAssignments: [[0, 1, 2]])
    let witness = [frFromInt(3), frFromInt(5), frFromInt(99)]

    let wireEvals = engine.assignWires(circuit: circuit, witness: witness)
    let result = engine.evaluateGateConstraints(circuit: circuit, wireEvals: wireEvals)

    expect(!result.allSatisfied, "Gate violated: 3 + 5 != 99")
    expectEqual(result.failingRows.count, 1, "Exactly 1 failing row")
    expectEqual(result.failingRows[0], 0, "Row 0 fails")
}

private func testGateConstraintMulGate() {
    let engine = GPUPlonkWireAssignEngine()

    // a * b - c = 0 with a=4, b=7, c=28
    let negOne = frSub(Fr.zero, Fr.one)
    let gate = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: negOne, qM: Fr.one, qC: Fr.zero)
    let circuit = PlonkCircuit(gates: [gate], copyConstraints: [],
                               wireAssignments: [[0, 1, 2]])
    let witness = [frFromInt(4), frFromInt(7), frFromInt(28)]

    let wireEvals = engine.assignWires(circuit: circuit, witness: witness)
    let result = engine.evaluateGateConstraints(circuit: circuit, wireEvals: wireEvals)

    expect(result.allSatisfied, "Mul gate a*b - c = 0 satisfied with (4, 7, 28)")
}

// MARK: - Gate Constraint with Rotation

private func testGateConstraintWithRotation() {
    let engine = GPUPlonkWireAssignEngine()

    // Two gates: row 0 and row 1
    // Gate 0: a + b - c = 0 (a=2, b=3, c=5)
    // Gate 1: a + b - c = 0 (a=5, b=1, c=6)  <- a[1] should equal c[0] (chaining)
    let negOne = frSub(Fr.zero, Fr.one)
    let gate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)
    let circuit = PlonkCircuit(
        gates: [gate, gate],
        copyConstraints: [],
        wireAssignments: [[0, 1, 2], [2, 3, 4]]
    )
    let witness = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(1), frFromInt(6)]

    let wireEvals = engine.assignWires(circuit: circuit, witness: witness)
    let rotatedEvals = engine.computeRotations(evaluations: wireEvals)

    // qNext = [1, 0, 0, 0] -> enforces w0_next[0] == w2[0] i.e. a[1] == c[0]
    let qNext = [Fr.one, Fr.zero, Fr.zero, Fr.zero]

    let result = engine.evaluateGateConstraintsWithRotation(
        circuit: circuit,
        wireEvals: wireEvals,
        rotatedEvals: rotatedEvals,
        qNext: qNext
    )

    // w0_next[0] = wireEvals[0][1] = 5 and w2[0] = wireEvals[2][0] = 5
    // So rotation constraint is 1 * (5 - 5) = 0. Base constraint also 0.
    expect(result.allSatisfied, "Gate constraint with rotation satisfied (chained output)")
}

// MARK: - Polynomial Evaluation at Point

private func testPolynomialEvalAtPoint() {
    let engine = GPUPlonkWireAssignEngine()

    // Polynomial p(x) = 3 + 2*x + x^2
    // p(5) = 3 + 10 + 25 = 38
    let coeffs: [[Fr]] = [[frFromInt(3), frFromInt(2), frFromInt(1), Fr.zero]]
    let zeta = frFromInt(5)

    let result = engine.evaluateAtPoint(wireCoeffs: coeffs, zeta: zeta)

    expect(frEqual(result.evals[0], frFromInt(38)), "p(5) = 38 for p(x) = 3 + 2x + x^2")
    expect(result.shiftedEvals == nil, "No shifted evals when not requested")
}

private func testPolynomialEvalShifted() {
    let engine = GPUPlonkWireAssignEngine()

    // p(x) = 1 + x
    // p(zeta) for zeta=3: 1 + 3 = 4
    // p(zeta * omega) for omega=2: 1 + 6 = 7
    let coeffs: [[Fr]] = [[Fr.one, Fr.one, Fr.zero, Fr.zero]]
    let zeta = frFromInt(3)
    let omega = frFromInt(2)

    let result = engine.evaluateAtPoint(
        wireCoeffs: coeffs,
        zeta: zeta,
        omega: omega,
        includeShifted: true
    )

    expect(frEqual(result.evals[0], frFromInt(4)), "p(3) = 4 for p(x) = 1 + x")
    expect(result.shiftedEvals != nil, "Shifted evals computed")
    expect(frEqual(result.shiftedEvals![0], frFromInt(7)), "p(6) = 7 for p(x) = 1 + x")
}

private func testHornerEvalConstant() {
    let engine = GPUPlonkWireAssignEngine()

    // Constant polynomial p(x) = 42
    let coeffs: [[Fr]] = [[frFromInt(42), Fr.zero, Fr.zero, Fr.zero]]
    let zeta = frFromInt(999)

    let result = engine.evaluateAtPoint(wireCoeffs: coeffs, zeta: zeta)

    expect(frEqual(result.evals[0], frFromInt(42)), "Constant poly evaluates to 42 everywhere")
}

// MARK: - Vanishing Polynomial

private func testVanishingPolyEval() {
    let engine = GPUPlonkWireAssignEngine()

    // Z_H(x) = x^n - 1. For n=4 and x=omega (4th root of unity):
    // Z_H(omega) = omega^4 - 1 = 1 - 1 = 0
    let omega = frRootOfUnity(logN: 2) // 4th root of unity
    let zh = engine.vanishingPolyEval(point: omega, domainSize: 4)
    expect(zh.isZero, "Z_H(omega) = 0 for omega a 4th root of unity")

    // Z_H(2) = 2^4 - 1 = 15 for n=4
    let zh2 = engine.vanishingPolyEval(point: frFromInt(2), domainSize: 4)
    expect(frEqual(zh2, frFromInt(15)), "Z_H(2) = 15 for domain size 4")
}

// MARK: - Lagrange Basis

private func testLagrangeBasisEval() {
    let engine = GPUPlonkWireAssignEngine()

    // L_0(omega^0) = L_0(1) = 1
    let omega = frRootOfUnity(logN: 2) // 4th root
    let l0At1 = engine.lagrangeBasisEval(index: 0, zeta: Fr.one, domainSize: 4, omega: omega)
    expect(frEqual(l0At1, Fr.one), "L_0(1) = 1")

    // L_0(omega) = 0 (Lagrange basis is 0 at other roots)
    let l0AtOmega = engine.lagrangeBasisEval(index: 0, zeta: omega, domainSize: 4, omega: omega)
    expect(l0AtOmega.isZero, "L_0(omega) = 0")

    // L_1(omega) = 1
    let l1AtOmega = engine.lagrangeBasisEval(index: 1, zeta: omega, domainSize: 4, omega: omega)
    expect(frEqual(l1AtOmega, Fr.one), "L_1(omega) = 1")
}

// MARK: - Batch NTT Round-Trip

private func testBatchNTTRoundTrip() {
    do {
        let engine = GPUPlonkWireAssignEngine()
        let ntt = try NTTEngine()

        // Wire evaluations
        let evals: [[Fr]] = [
            [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)],
            [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)],
        ]

        // Forward NTT then inverse NTT should recover original
        let forward = try engine.batchNTT(wireData: evals, ntt: ntt, inverse: false)
        let recovered = try engine.batchNTT(wireData: forward, ntt: ntt, inverse: true)

        expectEqual(recovered.count, 2, "2 wire columns recovered")
        for w in 0..<2 {
            for i in 0..<4 {
                expect(frEqual(recovered[w][i], evals[w][i]),
                       "Batch NTT round-trip: wire \(w) row \(i)")
            }
        }
    } catch {
        expect(false, "Batch NTT round-trip threw: \(error)")
    }
}

// MARK: - Wire Blinding

private func testWireBlinding() {
    let engine = GPUPlonkWireAssignEngine()

    // 4-element coefficient polynomial, blinding degree 2
    let wireCoeffs: [[Fr]] = [
        [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    ]
    let blindingFactors: [[Fr]] = [
        [frFromInt(100), frFromInt(200)]
    ]

    let blinded = engine.addBlinding(
        wireCoeffs: wireCoeffs,
        blindingFactors: blindingFactors,
        blindingDegree: 2
    )

    expectEqual(blinded.count, 1, "1 wire column blinded")
    // Indices 0, 1 unchanged; indices 2, 3 have blinding added
    expect(frEqual(blinded[0][0], frFromInt(1)), "Blinding: coeff 0 unchanged")
    expect(frEqual(blinded[0][1], frFromInt(2)), "Blinding: coeff 1 unchanged")
    // coeff 2 = 3 + 100 = 103
    expect(frEqual(blinded[0][2], frFromInt(103)), "Blinding: coeff 2 = 3 + 100")
    // coeff 3 = 4 + 200 = 204
    expect(frEqual(blinded[0][3], frFromInt(204)), "Blinding: coeff 3 = 4 + 200")
}

// MARK: - Coset Evaluation

private func testCosetEvaluation() {
    do {
        let engine = GPUPlonkWireAssignEngine()
        let ntt = try NTTEngine()

        // p(x) = 1 + 0*x + 0*x^2 + 0*x^3 (constant polynomial)
        let wireCoeffs: [[Fr]] = [
            [Fr.one, Fr.zero, Fr.zero, Fr.zero]
        ]
        let cosetGen = frFromInt(5)

        let cosetEvals = try engine.evaluateOnCoset(
            wireCoeffs: wireCoeffs,
            cosetGen: cosetGen,
            ntt: ntt
        )

        expectEqual(cosetEvals.count, 1, "1 wire column on coset")
        expectEqual(cosetEvals[0].count, 4, "Domain size 4 on coset")
        // Constant polynomial evaluates to 1 everywhere
        for i in 0..<4 {
            expect(frEqual(cosetEvals[0][i], Fr.one),
                   "Constant poly on coset: eval[\(i)] = 1")
        }
    } catch {
        expect(false, "Coset evaluation threw: \(error)")
    }
}

// MARK: - Wire Commitment

private func testWireCommitment() {
    let engine = GPUPlonkWireAssignEngine()

    // Trivial SRS with 4 points (just use a basic generator for testing)
    // We use the identity for simplicity — the commitment of an all-zero
    // polynomial should be the identity point
    let wireCoeffs: [[Fr]] = [
        [Fr.zero, Fr.zero, Fr.zero, Fr.zero]
    ]
    // Create minimal SRS (will give identity commitment for zero coeffs)
    let srs = [PointAffine(x: Fp.one, y: Fp.one)]

    let commitments = engine.commitWirePolynomials(wireCoeffs: wireCoeffs, srs: srs)

    expectEqual(commitments.numWires, 1, "1 wire committed")
    expectEqual(commitments.commitments.count, 1, "1 commitment")
    // Zero polynomial -> identity commitment
    expect(pointIsIdentity(commitments.commitments[0]),
           "Zero polynomial commits to identity point")
    expect(commitments.rotatedCommitments == nil, "No rotated commitments by default")
}

// MARK: - Empty Circuit

private func testEmptyCircuit() {
    let engine = GPUPlonkWireAssignEngine()

    let circuit = PlonkCircuit(gates: [], copyConstraints: [], wireAssignments: [])
    let witness: [Fr] = []

    let wireEvals = engine.assignWires(circuit: circuit, witness: witness)
    // Even empty circuit gets padded to minimum domain size
    expect(wireEvals[0].count >= 4, "Empty circuit still has minimum domain size")

    let result = engine.evaluateGateConstraints(circuit: circuit, wireEvals: wireEvals)
    expect(result.allSatisfied, "Empty circuit satisfies all constraints trivially")
    expectEqual(result.failingRows.count, 0, "Empty circuit has 0 failing rows")
}

// MARK: - Config Defaults

private func testConfigDefaults() {
    let config = WireAssignConfig()

    expectEqual(config.numWires, 3, "Default: 3 wires (standard Plonk)")
    expect(!config.computeRotations, "Default: rotations not computed")
    expect(config.publicInputIndices.isEmpty, "Default: no public inputs")
    expectEqual(config.minDomainSize, 4, "Default: minimum domain size 4")
}

// MARK: - Multi-Gate Circuit

private func testMultiGateCircuit() {
    let engine = GPUPlonkWireAssignEngine()

    let negOne = frSub(Fr.zero, Fr.one)
    // Gate 0: a + b - c = 0 (addition, a=2, b=3, c=5)
    let addGate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)
    // Gate 1: a * b - c = 0 (multiplication, a=4, b=6, c=24)
    let mulGate = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: negOne, qM: Fr.one, qC: Fr.zero)
    // Gate 2: a + 0 + 0 + 0 + (-5) = 0 (constant check, a=5)
    let constGate = PlonkGate(qL: Fr.one, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero,
                               qC: frNeg(frFromInt(5)))
    // Gate 3: zero gate (always satisfied)
    let zeroGate = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: Fr.zero, qM: Fr.zero, qC: Fr.zero)

    let circuit = PlonkCircuit(
        gates: [addGate, mulGate, constGate, zeroGate],
        copyConstraints: [],
        wireAssignments: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    )
    let witness = [
        frFromInt(2), frFromInt(3), frFromInt(5),    // gate 0
        frFromInt(4), frFromInt(6), frFromInt(24),   // gate 1
        frFromInt(5), Fr.zero, Fr.zero,               // gate 2
        Fr.zero, Fr.zero, Fr.zero                     // gate 3
    ]

    let wireEvals = engine.assignWires(circuit: circuit, witness: witness)
    let result = engine.evaluateGateConstraints(circuit: circuit, wireEvals: wireEvals)

    expect(result.allSatisfied, "Multi-gate circuit: all 4 gates satisfied")
    expectEqual(result.failingRows.count, 0, "Multi-gate circuit: 0 failing rows")
    expectEqual(result.rowCount, 4, "Multi-gate circuit: domain size 4")
}

// MARK: - Larger Circuit Stress Test

private func testWireAssignLargerCircuit() {
    let engine = GPUPlonkWireAssignEngine()

    // 16 addition gates: row i computes i + (i+1) = (2i+1)
    let negOne = frSub(Fr.zero, Fr.one)
    let gate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)

    var gates = [PlonkGate]()
    var wireAssigns = [[Int]]()
    var witness = [Fr]()

    for i in 0..<16 {
        gates.append(gate)
        let base = i * 3
        wireAssigns.append([base, base + 1, base + 2])
        witness.append(frFromInt(UInt64(i)))
        witness.append(frFromInt(UInt64(i + 1)))
        witness.append(frFromInt(UInt64(2 * i + 1)))
    }

    let circuit = PlonkCircuit(gates: gates, copyConstraints: [], wireAssignments: wireAssigns)
    let wireEvals = engine.assignWires(circuit: circuit, witness: witness)

    expectEqual(wireEvals[0].count, 16, "16 gates fits in domain size 16")

    let result = engine.evaluateGateConstraints(circuit: circuit, wireEvals: wireEvals)
    expect(result.allSatisfied, "16-gate addition circuit fully satisfied")

    // Verify specific wire values
    expect(frEqual(wireEvals[0][0], frFromInt(0)), "Larger circuit: a[0] = 0")
    expect(frEqual(wireEvals[1][0], frFromInt(1)), "Larger circuit: b[0] = 1")
    expect(frEqual(wireEvals[2][0], frFromInt(1)), "Larger circuit: c[0] = 1")
    expect(frEqual(wireEvals[0][15], frFromInt(15)), "Larger circuit: a[15] = 15")
    expect(frEqual(wireEvals[2][15], frFromInt(31)), "Larger circuit: c[15] = 31")
}

// MARK: - Quotient Wire Contribution

private func testQuotientWireContribution() {
    let engine = GPUPlonkWireAssignEngine()

    // Simple satisfied circuit: a + b - c = 0 across 4 rows
    // When all constraints are satisfied, quotient numerator should be zero
    let negOne = frSub(Fr.zero, Fr.one)
    let gate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)

    // 4 gates all satisfied
    let circuit = PlonkCircuit(
        gates: [gate, gate, gate, gate],
        copyConstraints: [],
        wireAssignments: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
    )
    let witness = [
        frFromInt(1), frFromInt(2), frFromInt(3),
        frFromInt(4), frFromInt(5), frFromInt(9),
        frFromInt(10), frFromInt(20), frFromInt(30),
        frFromInt(7), frFromInt(8), frFromInt(15),
    ]

    let wireEvals = engine.assignWires(circuit: circuit, witness: witness)
    let cosetGen = frFromInt(5)

    // For satisfied constraints, the numerator at the actual gate rows should be zero.
    // The quotient computation evaluates numerator / Z_H on a coset, so
    // we just check the computation completes without error.
    let quotient = engine.computeQuotientWireContribution(
        circuit: circuit,
        wireCosetEvals: wireEvals, // using eval domain instead of coset for simplicity
        cosetGen: cosetGen,
        domainSize: 4
    )

    expectEqual(quotient.count, 4, "Quotient has domain-size entries")
    // The gate constraints evaluate to zero at the actual gate rows,
    // so quotient should be zero at those positions (since numerator is 0)
    // Note: wireEvals are not on the coset, so the Z_H division may be non-trivial,
    // but the numerator IS zero for the first 4 gates, so the quotient is 0/Z_H = 0.
    for i in 0..<4 {
        // Numerator is qL*a + qR*b + qO*c + qC = 0 for satisfied gates
        // So quotient[i] = 0 / Z_H(cosetPoint) = 0
        expect(quotient[i].isZero,
               "Quotient contribution is zero for satisfied gate at row \(i)")
    }
}

// MARK: - Wire Polynomial Computation Integration

private func testWirePolyComputation() {
    do {
        let engine = GPUPlonkWireAssignEngine()
        let ntt = try NTTEngine()

        let negOne = frSub(Fr.zero, Fr.one)
        let gate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)
        let circuit = PlonkCircuit(
            gates: [gate],
            copyConstraints: [],
            wireAssignments: [[0, 1, 2]]
        )
        let witness = [frFromInt(3), frFromInt(5), frFromInt(8)]

        let config = WireAssignConfig(computeRotations: true)
        let polySet = try engine.computeWirePolynomials(
            circuit: circuit,
            witness: witness,
            config: config,
            ntt: ntt
        )

        expectEqual(polySet.numWires, 3, "3 wire polynomials")
        expect(polySet.domainSize >= 4, "Domain size >= 4")
        expect(polySet.rotated != nil, "Rotations computed when requested")
        expectEqual(polySet.coefficients.count, 3, "3 coefficient arrays")

        // Verify round-trip: NTT(coefficients) should recover evaluations
        let recoveredEvals = try engine.batchNTT(wireData: polySet.coefficients, ntt: ntt,
                                                  inverse: false)
        for w in 0..<3 {
            for i in 0..<polySet.domainSize {
                expect(frEqual(recoveredEvals[w][i], polySet.evaluations[w][i]),
                       "Wire poly round-trip: wire \(w) row \(i)")
            }
        }
    } catch {
        expect(false, "Wire polynomial computation threw: \(error)")
    }
}
