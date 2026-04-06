// GPUConstraintSatEngine Tests — GPU-accelerated constraint satisfiability checker
//
// Tests R1CS, Plonkish, and AIR constraint satisfaction checking with valid
// and invalid witnesses, public input injection, constraint degree analysis,
// gate descriptor checking, batch checking, and edge cases.

import Foundation
@testable import zkMetal

public func runGPUConstraintSatTests() {
    suite("GPUConstraintSat")

    testR1CSSatisfied()
    testR1CSUnsatisfied()
    testR1CSPublicInputInjection()
    testR1CSBatchCheck()
    testR1CSDimensionMismatch()
    testR1CSEmptySystem()
    testR1CSFromSystem()
    testPlonkishSatisfied()
    testPlonkishUnsatisfied()
    testPlonkishMulGate()
    testPlonkishBoolGate()
    testPlonkishPublicInputInjection()
    testPlonkishBatchCheck()
    testPlonkishEmptyCircuit()
    testPlonkishMultipleGates()
    testAIRSatisfied()
    testAIRUnsatisfied()
    testAIRBoundaryConstraint()
    testAIRBoundaryViolation()
    testAIREmptyTrace()
    testAIRMultipleTransitions()
    testAIRFibonacci()
    testConstraintSystemSatisfied()
    testConstraintSystemUnsatisfied()
    testConstraintSystemCrossRow()
    testGateDescriptorMulSatisfied()
    testGateDescriptorMulUnsatisfied()
    testGateDescriptorAddSatisfied()
    testGateDescriptorBoolSatisfied()
    testGateDescriptorBoolViolation()
    testGateDescriptorArithmetic()
    testGateDescriptorRangeDecomp()
    testGateDescriptorWithSelector()
    testDegreeAnalysis()
    testDegreeCheck()
    testDegreePow()
    testSatisfactionResultSummary()
    testViolationDescription()
    testInjectPublicInputs()
    testLargeR1CS()
    testLargePlonkish()
    testLargeAIR()
}

// MARK: - R1CS Tests

private func testR1CSSatisfied() {
    let engine = GPUConstraintSatEngine()

    // Simple R1CS: one constraint a * b = c
    // z = [1, a, b, c] = [1, 3, 5, 15]
    let A = buildSparseMatrix(rows: 1, cols: 4, entries: [(0, 1, Fr.one)])
    let B = buildSparseMatrix(rows: 1, cols: 4, entries: [(0, 2, Fr.one)])
    let C = buildSparseMatrix(rows: 1, cols: 4, entries: [(0, 3, Fr.one)])

    let z = [Fr.one, frFromInt(3), frFromInt(5), frFromInt(15)]
    let input = R1CSSatInput(A: A, B: B, C: C, z: z, numPublicInputs: 0)

    let result = engine.checkR1CS(input)
    expect(result.isSatisfied, "R1CS satisfied: 3 * 5 = 15")
    expect(result.firstViolation == nil, "R1CS satisfied: no violation")
    expectEqual(result.numConstraints, 1, "R1CS: 1 constraint")
    print("  [OK] R1CS satisfied (3*5=15)")
}

private func testR1CSUnsatisfied() {
    let engine = GPUConstraintSatEngine()

    // a * b = c, but z has wrong c
    let A = buildSparseMatrix(rows: 1, cols: 4, entries: [(0, 1, Fr.one)])
    let B = buildSparseMatrix(rows: 1, cols: 4, entries: [(0, 2, Fr.one)])
    let C = buildSparseMatrix(rows: 1, cols: 4, entries: [(0, 3, Fr.one)])

    let z = [Fr.one, frFromInt(3), frFromInt(5), frFromInt(10)]  // 3*5 != 10
    let input = R1CSSatInput(A: A, B: B, C: C, z: z, numPublicInputs: 0,
                             labels: ["mul_gate_0"])

    let result = engine.checkR1CS(input)
    expect(!result.isSatisfied, "R1CS unsatisfied: 3*5 != 10")
    expect(result.firstViolation != nil, "R1CS unsatisfied: has violation")
    if let v = result.firstViolation {
        expectEqual(v.constraintIndex, 0, "R1CS: violation at constraint 0")
        expectEqual(v.label, "mul_gate_0", "R1CS: violation label")
        expect(!v.residual.isZero, "R1CS: non-zero residual")
    }
    print("  [OK] R1CS unsatisfied (3*5 != 10)")
}

private func testR1CSPublicInputInjection() {
    let engine = GPUConstraintSatEngine()

    // a * b = c, public input a=4
    let A = buildSparseMatrix(rows: 1, cols: 4, entries: [(0, 1, Fr.one)])
    let B = buildSparseMatrix(rows: 1, cols: 4, entries: [(0, 2, Fr.one)])
    let C = buildSparseMatrix(rows: 1, cols: 4, entries: [(0, 3, Fr.one)])

    // z starts wrong for a, but we inject public input
    let z = [Fr.one, frFromInt(0), frFromInt(7), frFromInt(28)]  // 0*7 != 28
    let input = R1CSSatInput(A: A, B: B, C: C, z: z, numPublicInputs: 1)

    // Inject a=4 at index 1 => 4*7=28
    let bindings = [PublicInputBinding(variableIndex: 1, value: frFromInt(4))]
    let result = engine.checkR1CS(input, publicInputs: bindings)
    expect(result.isSatisfied, "R1CS public input injection: 4*7=28")
    print("  [OK] R1CS public input injection")
}

private func testR1CSBatchCheck() {
    let engine = GPUConstraintSatEngine()

    let A = buildSparseMatrix(rows: 1, cols: 4, entries: [(0, 1, Fr.one)])
    let B = buildSparseMatrix(rows: 1, cols: 4, entries: [(0, 2, Fr.one)])
    let C = buildSparseMatrix(rows: 1, cols: 4, entries: [(0, 3, Fr.one)])

    let system = R1CSSystem(A: A, B: B, C: C, numPublicInputs: 0)

    let witnesses: [[Fr]] = [
        [Fr.one, frFromInt(2), frFromInt(3), frFromInt(6)],   // 2*3=6 OK
        [Fr.one, frFromInt(4), frFromInt(5), frFromInt(20)],  // 4*5=20 OK
        [Fr.one, frFromInt(7), frFromInt(8), frFromInt(55)],  // 7*8=55 WRONG (should be 56)
    ]

    let results = engine.batchCheckR1CS(system: system, witnesses: witnesses)
    expectEqual(results.count, 3, "batch R1CS: 3 results")
    expect(results[0].isSatisfied, "batch R1CS: witness 0 satisfied")
    expect(results[1].isSatisfied, "batch R1CS: witness 1 satisfied")
    expect(!results[2].isSatisfied, "batch R1CS: witness 2 unsatisfied")
    print("  [OK] R1CS batch check (2 satisfied, 1 unsatisfied)")
}

private func testR1CSDimensionMismatch() {
    let engine = GPUConstraintSatEngine()

    let A = buildSparseMatrix(rows: 1, cols: 4, entries: [(0, 1, Fr.one)])
    let B = buildSparseMatrix(rows: 1, cols: 4, entries: [(0, 2, Fr.one)])
    let C = buildSparseMatrix(rows: 1, cols: 4, entries: [(0, 3, Fr.one)])

    // z has wrong length (3 instead of 4)
    let z = [Fr.one, frFromInt(2), frFromInt(3)]
    let input = R1CSSatInput(A: A, B: B, C: C, z: z, numPublicInputs: 0)

    let result = engine.checkR1CS(input)
    expect(!result.isSatisfied, "R1CS dimension mismatch: should fail")
    print("  [OK] R1CS dimension mismatch")
}

private func testR1CSEmptySystem() {
    let engine = GPUConstraintSatEngine()

    let A = buildSparseMatrix(rows: 0, cols: 1, entries: [])
    let B = buildSparseMatrix(rows: 0, cols: 1, entries: [])
    let C = buildSparseMatrix(rows: 0, cols: 1, entries: [])

    let z = [Fr.one]
    let input = R1CSSatInput(A: A, B: B, C: C, z: z, numPublicInputs: 0)

    let result = engine.checkR1CS(input)
    expect(result.isSatisfied, "R1CS empty system: vacuously satisfied")
    expectEqual(result.numConstraints, 0, "R1CS empty: 0 constraints")
    print("  [OK] R1CS empty system")
}

private func testR1CSFromSystem() {
    let engine = GPUConstraintSatEngine()
    let A = buildSparseMatrix(rows: 2, cols: 5, entries: [(0, 1, Fr.one), (1, 1, Fr.one), (1, 3, Fr.one)])
    let B = buildSparseMatrix(rows: 2, cols: 5, entries: [(0, 2, Fr.one), (1, 0, Fr.one)])
    let C = buildSparseMatrix(rows: 2, cols: 5, entries: [(0, 3, Fr.one), (1, 4, Fr.one)])
    let system = R1CSSystem(A: A, B: B, C: C, numPublicInputs: 1)
    let z = [Fr.one, frFromInt(3), frFromInt(5), frFromInt(15), frFromInt(18)]
    let result = engine.checkR1CS(R1CSSatInput(system: system, z: z))
    expect(result.isSatisfied, "R1CS from system: 2 constraints satisfied")
    expectEqual(result.numConstraints, 2, "R1CS from system: 2 constraints")
    print("  [OK] R1CS from R1CSSystem (2 constraints)")
}

// MARK: - Plonkish Tests

private func testPlonkishSatisfied() {
    let engine = GPUConstraintSatEngine()

    // a + b - c = 0 => qL=1, qR=1, qO=-1, qM=0, qC=0
    let negOne = frSub(Fr.zero, Fr.one)
    let gate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)

    let witness: [Int: Fr] = [0: frFromInt(3), 1: frFromInt(7), 2: frFromInt(10)]
    let input = PlonkishSatInput(gates: [gate], wireAssignments: [[0, 1, 2]],
                                 witness: witness)

    let result = engine.checkPlonkish(input)
    expect(result.isSatisfied, "Plonkish satisfied: 3+7=10")
    print("  [OK] Plonkish satisfied (3+7=10)")
}

private func testPlonkishUnsatisfied() {
    let engine = GPUConstraintSatEngine()

    let negOne = frSub(Fr.zero, Fr.one)
    let gate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)

    let witness: [Int: Fr] = [0: frFromInt(3), 1: frFromInt(7), 2: frFromInt(11)]  // 3+7 != 11
    let input = PlonkishSatInput(gates: [gate], wireAssignments: [[0, 1, 2]],
                                 witness: witness)

    let result = engine.checkPlonkish(input)
    expect(!result.isSatisfied, "Plonkish unsatisfied: 3+7 != 11")
    if let v = result.firstViolation {
        expectEqual(v.constraintIndex, 0, "Plonkish: violation at gate 0")
    }
    print("  [OK] Plonkish unsatisfied (3+7 != 11)")
}

private func testPlonkishMulGate() {
    let engine = GPUConstraintSatEngine()

    // a*b - c = 0 => qL=0, qR=0, qO=-1, qM=1, qC=0
    let negOne = frSub(Fr.zero, Fr.one)
    let gate = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: negOne, qM: Fr.one, qC: Fr.zero)

    let witness: [Int: Fr] = [0: frFromInt(6), 1: frFromInt(7), 2: frFromInt(42)]
    let input = PlonkishSatInput(gates: [gate], wireAssignments: [[0, 1, 2]],
                                 witness: witness)

    let result = engine.checkPlonkish(input)
    expect(result.isSatisfied, "Plonkish mul gate: 6*7=42")
    print("  [OK] Plonkish mul gate (6*7=42)")
}

private func testPlonkishBoolGate() {
    let engine = GPUConstraintSatEngine()

    // a*(1-a)=0 => qL=1, qR=0, qO=0, qM=-1, qC=0
    // Actually: qL*a + qM*a*b = 0, where b=a
    // a - a^2 = 0 => qL=1, qM=-1 with a=b
    let negOne = frSub(Fr.zero, Fr.one)
    let gate = PlonkGate(qL: Fr.one, qR: Fr.zero, qO: Fr.zero, qM: negOne, qC: Fr.zero)

    // a=1, b=1 (same wire): 1*1 + (-1)*1*1 = 1 - 1 = 0
    let witness0: [Int: Fr] = [0: Fr.one]
    let input0 = PlonkishSatInput(gates: [gate], wireAssignments: [[0, 0, 0]],
                                  witness: witness0)
    let result0 = engine.checkPlonkish(input0)
    expect(result0.isSatisfied, "Plonkish bool gate: a=1 satisfied")

    // a=0: 1*0 + (-1)*0*0 = 0
    let witness1: [Int: Fr] = [0: Fr.zero]
    let input1 = PlonkishSatInput(gates: [gate], wireAssignments: [[0, 0, 0]],
                                  witness: witness1)
    let result1 = engine.checkPlonkish(input1)
    expect(result1.isSatisfied, "Plonkish bool gate: a=0 satisfied")

    // a=2: 1*2 + (-1)*2*2 = 2 - 4 = -2 != 0
    let witness2: [Int: Fr] = [0: frFromInt(2)]
    let input2 = PlonkishSatInput(gates: [gate], wireAssignments: [[0, 0, 0]],
                                  witness: witness2)
    let result2 = engine.checkPlonkish(input2)
    expect(!result2.isSatisfied, "Plonkish bool gate: a=2 unsatisfied")

    print("  [OK] Plonkish bool gate")
}

private func testPlonkishPublicInputInjection() {
    let engine = GPUConstraintSatEngine()

    let negOne = frSub(Fr.zero, Fr.one)
    let gate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)

    // Start with wrong witness for a, inject correct value
    let witness: [Int: Fr] = [0: frFromInt(99), 1: frFromInt(5), 2: frFromInt(11)]
    let input = PlonkishSatInput(gates: [gate], wireAssignments: [[0, 1, 2]],
                                 witness: witness, publicInputIndices: [0])

    // Inject a=6 => 6+5=11
    let bindings = [PublicInputBinding(variableIndex: 0, value: frFromInt(6))]
    let result = engine.checkPlonkish(input, publicInputs: bindings)
    expect(result.isSatisfied, "Plonkish public input injection: 6+5=11")
    print("  [OK] Plonkish public input injection")
}

private func testPlonkishBatchCheck() {
    let engine = GPUConstraintSatEngine()

    let negOne = frSub(Fr.zero, Fr.one)
    let gate = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: negOne, qM: Fr.one, qC: Fr.zero)

    let circuit = PlonkCircuit(gates: [gate], copyConstraints: [],
                               wireAssignments: [[0, 1, 2]])

    let witnesses: [[Int: Fr]] = [
        [0: frFromInt(3), 1: frFromInt(4), 2: frFromInt(12)],  // 3*4=12 OK
        [0: frFromInt(5), 1: frFromInt(6), 2: frFromInt(30)],  // 5*6=30 OK
        [0: frFromInt(2), 1: frFromInt(9), 2: frFromInt(17)],  // 2*9=17 WRONG
    ]

    let results = engine.batchCheckPlonkish(circuit: circuit, witnesses: witnesses)
    expectEqual(results.count, 3, "batch Plonkish: 3 results")
    expect(results[0].isSatisfied, "batch Plonkish: witness 0 OK")
    expect(results[1].isSatisfied, "batch Plonkish: witness 1 OK")
    expect(!results[2].isSatisfied, "batch Plonkish: witness 2 FAIL")
    print("  [OK] Plonkish batch check")
}

private func testPlonkishEmptyCircuit() {
    let engine = GPUConstraintSatEngine()

    let input = PlonkishSatInput(gates: [], wireAssignments: [], witness: [:])
    let result = engine.checkPlonkish(input)
    expect(result.isSatisfied, "Plonkish empty circuit: vacuously satisfied")
    expectEqual(result.numConstraints, 0, "Plonkish empty: 0 constraints")
    print("  [OK] Plonkish empty circuit")
}

private func testPlonkishMultipleGates() {
    let engine = GPUConstraintSatEngine()
    let negOne = frSub(Fr.zero, Fr.one)
    let addGate = PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero)
    let mulGate = PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: negOne, qM: Fr.one, qC: Fr.zero)
    let a = frFromInt(3), b = frFromInt(5)
    let c = frAdd(a, b), d = frMul(a, b), e = frAdd(c, d)
    let witness: [Int: Fr] = [0: a, 1: b, 2: c, 3: d, 4: e]
    let input = PlonkishSatInput(
        gates: [addGate, mulGate, addGate],
        wireAssignments: [[0, 1, 2], [0, 1, 3], [2, 3, 4]], witness: witness)
    let result = engine.checkPlonkish(input)
    expect(result.isSatisfied, "Plonkish multiple gates: chain satisfied")
    expectEqual(result.numConstraints, 3, "Plonkish: 3 gates")
    print("  [OK] Plonkish multiple gates (add+mul+chain)")
}

// MARK: - AIR Tests

private func testAIRSatisfied() {
    let engine = GPUConstraintSatEngine()

    // Simple AIR: col[0] at next row = col[0] at current row + 1
    // Transition: next(0) - col(0) - 1 = 0
    let numRows = 8
    var col0 = [Fr]()
    for i in 0..<numRows {
        col0.append(frFromInt(UInt64(i * 3)))  // 0, 3, 6, 9, ...
    }

    // Transition: next(0) - col(0) - 3 = 0
    let three = frFromInt(3)
    let transition: Expr = .wire(Wire(index: 0, row: 1)) - .wire(Wire.col(0)) - .constant(three)

    let input = AIRSatInput(trace: [col0], transitionConstraints: [transition])
    let result = engine.checkAIR(input)
    expect(result.isSatisfied, "AIR satisfied: linear increment by 3")
    print("  [OK] AIR satisfied (increment by 3)")
}

private func testAIRUnsatisfied() {
    let engine = GPUConstraintSatEngine()

    let numRows = 8
    var col0 = [Fr]()
    for i in 0..<numRows {
        col0.append(frFromInt(UInt64(i * 3)))
    }
    // Break the pattern at row 4
    col0[4] = frFromInt(999)

    let three = frFromInt(3)
    let transition: Expr = .wire(Wire(index: 0, row: 1)) - .wire(Wire.col(0)) - .constant(three)

    let input = AIRSatInput(trace: [col0], transitionConstraints: [transition])
    let result = engine.checkAIR(input)
    expect(!result.isSatisfied, "AIR unsatisfied: broken pattern")
    if let v = result.firstViolation {
        // Violation should be at row 3 (next row is 4 which is wrong) or row 4 (prev was wrong)
        expect(v.row >= 3 && v.row <= 4, "AIR: violation at row 3 or 4")
    }
    print("  [OK] AIR unsatisfied (broken pattern)")
}

private func testAIRBoundaryConstraint() {
    let engine = GPUConstraintSatEngine()

    let numRows = 4
    let col0: [Fr] = [frFromInt(10), frFromInt(13), frFromInt(16), frFromInt(19)]

    let three = frFromInt(3)
    let transition: Expr = .wire(Wire(index: 0, row: 1)) - .wire(Wire.col(0)) - .constant(three)

    let input = AIRSatInput(
        trace: [col0],
        transitionConstraints: [transition],
        boundaryConstraints: [(col: 0, row: 0, value: frFromInt(10))])

    let result = engine.checkAIR(input)
    expect(result.isSatisfied, "AIR boundary: col[0][0] = 10")
    print("  [OK] AIR boundary constraint satisfied")
}

private func testAIRBoundaryViolation() {
    let engine = GPUConstraintSatEngine()

    let col0: [Fr] = [frFromInt(10), frFromInt(13), frFromInt(16), frFromInt(19)]

    let three = frFromInt(3)
    let transition: Expr = .wire(Wire(index: 0, row: 1)) - .wire(Wire.col(0)) - .constant(three)

    // Expect first element to be 5, but it's 10
    let input = AIRSatInput(
        trace: [col0],
        transitionConstraints: [transition],
        boundaryConstraints: [(col: 0, row: 0, value: frFromInt(5))])

    let result = engine.checkAIR(input)
    expect(!result.isSatisfied, "AIR boundary violation: expected 5 got 10")
    if let v = result.firstViolation {
        expectEqual(v.row, 0, "AIR boundary violation at row 0")
    }
    print("  [OK] AIR boundary violation detected")
}

private func testAIREmptyTrace() {
    let engine = GPUConstraintSatEngine()

    let input = AIRSatInput(trace: [], transitionConstraints: [])
    let result = engine.checkAIR(input)
    expect(result.isSatisfied, "AIR empty: vacuously satisfied")
    print("  [OK] AIR empty trace")
}

private func testAIRMultipleTransitions() {
    let engine = GPUConstraintSatEngine()
    let numRows = 6
    var colX = [Fr](), colY = [Fr]()
    for i in 0..<numRows {
        let x = frFromInt(UInt64(i))
        colX.append(x); colY.append(frMul(x, x))
    }
    let t1: Expr = .wire(Wire(index: 0, row: 1)) - .wire(Wire.col(0)) - .constant(Fr.one)
    let t2: Expr = .wire(Wire.col(1)) - .mul(.wire(Wire.col(0)), .wire(Wire.col(0)))
    let input = AIRSatInput(trace: [colX, colY], transitionConstraints: [t1, t2],
                            transitionLabels: ["increment", "square"])
    let result = engine.checkAIR(input)
    expect(result.isSatisfied, "AIR multiple transitions: increment + square")
    print("  [OK] AIR multiple transitions (x+1, x^2)")
}

private func testAIRFibonacci() {
    let engine = GPUConstraintSatEngine()
    let numRows = 10
    var colA = [Fr](), colB = [Fr]()
    var a = Fr.one, b = Fr.one
    for _ in 0..<numRows {
        colA.append(a); colB.append(b)
        let na = b; let nb = frAdd(a, b); a = na; b = nb
    }
    let t1: Expr = .wire(Wire(index: 0, row: 1)) - .wire(Wire.col(1))
    let t2: Expr = .wire(Wire(index: 1, row: 1)) - .wire(Wire.col(0)) - .wire(Wire.col(1))
    let input = AIRSatInput(trace: [colA, colB], transitionConstraints: [t1, t2],
                            boundaryConstraints: [(col: 0, row: 0, value: Fr.one), (col: 1, row: 0, value: Fr.one)])
    let result = engine.checkAIR(input)
    expect(result.isSatisfied, "AIR Fibonacci: 10 rows satisfied")
    print("  [OK] AIR Fibonacci (10 rows)")
}

// MARK: - ConstraintSystem IR Tests

private func testConstraintSystemSatisfied() {
    let engine = GPUConstraintSatEngine()

    // a * b - c = 0
    let cs = ConstraintSystem(numWires: 3)
    cs.assertMul(Wire.col(0), Wire.col(1), Wire.col(2))

    let numRows = 4
    var colA = [Fr](), colB = [Fr](), colC = [Fr]()
    for i in 0..<numRows {
        let a = frFromInt(UInt64(i + 2))
        let b = frFromInt(UInt64(i + 3))
        colA.append(a)
        colB.append(b)
        colC.append(frMul(a, b))
    }

    let result = engine.checkConstraintSystem(cs, trace: [colA, colB, colC], numRows: numRows)
    expect(result.isSatisfied, "ConstraintSystem satisfied: a*b=c")
    print("  [OK] ConstraintSystem satisfied")
}

private func testConstraintSystemUnsatisfied() {
    let engine = GPUConstraintSatEngine()

    let cs = ConstraintSystem(numWires: 3)
    cs.assertMul(Wire.col(0), Wire.col(1), Wire.col(2))

    let numRows = 4
    var colA = [Fr](), colB = [Fr](), colC = [Fr]()
    for i in 0..<numRows {
        let a = frFromInt(UInt64(i + 2))
        let b = frFromInt(UInt64(i + 3))
        colA.append(a)
        colB.append(b)
        colC.append(frAdd(a, b))  // Wrong: c = a+b instead of a*b
    }

    let result = engine.checkConstraintSystem(cs, trace: [colA, colB, colC], numRows: numRows)
    expect(!result.isSatisfied, "ConstraintSystem unsatisfied: a*b != a+b")
    if let v = result.firstViolation {
        expectEqual(v.constraintIndex, 0, "CS: violation at constraint 0")
    }
    print("  [OK] ConstraintSystem unsatisfied")
}

private func testConstraintSystemCrossRow() {
    let engine = GPUConstraintSatEngine()

    // Constraint: next(0) = col(0) + 1
    let cs = ConstraintSystem(numWires: 1)
    cs.assertEqual(.wire(Wire.next(0)), .wire(Wire.col(0)) + .constant(Fr.one),
                   label: "increment")

    let numRows = 8
    var col0 = [Fr]()
    for i in 0..<numRows {
        col0.append(frFromInt(UInt64(i)))
    }

    let result = engine.checkConstraintSystem(cs, trace: [col0], numRows: numRows - 1)
    expect(result.isSatisfied, "ConstraintSystem cross-row: col[i+1]=col[i]+1")
    print("  [OK] ConstraintSystem cross-row constraint")
}

// MARK: - Gate Descriptor Tests

private func testGateDescriptorMulSatisfied() {
    let engine = GPUConstraintSatEngine()
    let domainSize = 16

    var colA = [Fr](), colB = [Fr](), colC = [Fr]()
    for i in 0..<domainSize {
        let a = frFromInt(UInt64(i + 1))
        let b = frFromInt(UInt64(i + 2))
        colA.append(a); colB.append(b); colC.append(frMul(a, b))
    }

    let gates = [GateDescriptor.mul(colA: 0, colB: 1, colC: 2)]
    let result = engine.checkGateDescriptors(trace: [colA, colB, colC], gates: gates,
                                              constants: [], domainSize: domainSize)
    expect(result.isSatisfied, "gate descriptor mul: satisfied")
    print("  [OK] gate descriptor mul satisfied")
}

private func testGateDescriptorMulUnsatisfied() {
    let engine = GPUConstraintSatEngine()
    let domainSize = 16

    var colA = [Fr](), colB = [Fr](), colC = [Fr]()
    for i in 0..<domainSize {
        let a = frFromInt(UInt64(i + 1))
        let b = frFromInt(UInt64(i + 2))
        colA.append(a); colB.append(b); colC.append(frAdd(a, b))  // Wrong: a+b instead of a*b
    }

    let gates = [GateDescriptor.mul(colA: 0, colB: 1, colC: 2)]
    let result = engine.checkGateDescriptors(trace: [colA, colB, colC], gates: gates,
                                              constants: [], domainSize: domainSize)
    expect(!result.isSatisfied, "gate descriptor mul: unsatisfied")
    if let v = result.firstViolation {
        expectEqual(v.label, "mul", "gate descriptor: mul label")
    }
    print("  [OK] gate descriptor mul unsatisfied")
}

private func testGateDescriptorAddSatisfied() {
    let engine = GPUConstraintSatEngine()
    let domainSize = 16

    var colA = [Fr](), colB = [Fr](), colC = [Fr]()
    for i in 0..<domainSize {
        let a = frFromInt(UInt64(i * 3 + 7))
        let b = frFromInt(UInt64(i * 5 + 11))
        colA.append(a); colB.append(b); colC.append(frAdd(a, b))
    }

    let gates = [GateDescriptor.add(colA: 0, colB: 1, colC: 2)]
    let result = engine.checkGateDescriptors(trace: [colA, colB, colC], gates: gates,
                                              constants: [], domainSize: domainSize)
    expect(result.isSatisfied, "gate descriptor add: satisfied")
    print("  [OK] gate descriptor add satisfied")
}

private func testGateDescriptorBoolSatisfied() {
    let engine = GPUConstraintSatEngine()
    let domainSize = 16

    var col = [Fr]()
    for i in 0..<domainSize {
        col.append(frFromInt(UInt64(i % 2)))  // alternating 0,1
    }

    let gates = [GateDescriptor.bool(col: 0)]
    let result = engine.checkGateDescriptors(trace: [col], gates: gates,
                                              constants: [], domainSize: domainSize)
    expect(result.isSatisfied, "gate descriptor bool: satisfied with 0/1 values")
    print("  [OK] gate descriptor bool satisfied")
}

private func testGateDescriptorBoolViolation() {
    let engine = GPUConstraintSatEngine()
    let domainSize = 8

    // Put a non-boolean value at row 3
    var col = [Fr]()
    for i in 0..<domainSize {
        if i == 3 {
            col.append(frFromInt(5))  // not boolean
        } else {
            col.append(frFromInt(UInt64(i % 2)))
        }
    }

    let gates = [GateDescriptor.bool(col: 0)]
    let result = engine.checkGateDescriptors(trace: [col], gates: gates,
                                              constants: [], domainSize: domainSize)
    expect(!result.isSatisfied, "gate descriptor bool: violation at non-boolean value")
    if let v = result.firstViolation {
        expectEqual(v.row, 3, "gate descriptor bool: violation at row 3")
    }
    print("  [OK] gate descriptor bool violation detected")
}

private func testGateDescriptorArithmetic() {
    let engine = GPUConstraintSatEngine()
    let domainSize = 16

    // 2*a + 3*b - c = 0
    let two = frFromInt(2)
    let three = frFromInt(3)
    let negOne = frSub(Fr.zero, Fr.one)
    let constants: [Fr] = [two, three, negOne, Fr.zero, Fr.zero]

    var colA = [Fr](), colB = [Fr](), colC = [Fr]()
    for i in 0..<domainSize {
        let a = frFromInt(UInt64(i + 1))
        let b = frFromInt(UInt64(i + 5))
        colA.append(a); colB.append(b)
        colC.append(frAdd(frMul(two, a), frMul(three, b)))
    }

    let gates = [GateDescriptor.arithmetic(colA: 0, colB: 1, colC: 2, constantsBaseIdx: 0)]
    let result = engine.checkGateDescriptors(trace: [colA, colB, colC], gates: gates,
                                              constants: constants, domainSize: domainSize)
    expect(result.isSatisfied, "gate descriptor arithmetic: 2a+3b=c")
    print("  [OK] gate descriptor arithmetic (2a+3b=c)")
}

private func testGateDescriptorRangeDecomp() {
    let engine = GPUConstraintSatEngine()
    let domainSize = 8
    let numBits = 4

    var colValue = [Fr]()
    var bitCols = [[Fr]](repeating: [Fr](), count: numBits)

    for i in 0..<domainSize {
        let val = UInt64(i)
        colValue.append(frFromInt(val))
        for b in 0..<numBits {
            let bit = (val >> UInt64(b)) & 1
            bitCols[b].append(frFromInt(bit))
        }
    }

    var trace = [colValue]
    trace.append(contentsOf: bitCols)

    let gates = [GateDescriptor.rangeDecomp(valueCol: 0, firstBitCol: 1, numBits: UInt32(numBits))]
    let result = engine.checkGateDescriptors(trace: trace, gates: gates,
                                              constants: [], domainSize: domainSize)
    expect(result.isSatisfied, "gate descriptor range decomp: 4-bit satisfied")
    print("  [OK] gate descriptor range decomposition")
}

private func testGateDescriptorWithSelector() {
    let engine = GPUConstraintSatEngine()
    let domainSize = 8

    // a * b - c = 0 with selector: only active on even rows
    var colA = [Fr](), colB = [Fr](), colC = [Fr]()
    var sel = [Fr]()
    for i in 0..<domainSize {
        let a = frFromInt(UInt64(i + 1))
        let b = frFromInt(UInt64(i + 2))
        colA.append(a); colB.append(b)
        if i % 2 == 0 {
            colC.append(frMul(a, b))  // correct on even rows
            sel.append(Fr.one)
        } else {
            colC.append(frFromInt(999))  // intentionally wrong on odd rows
            sel.append(Fr.zero)          // but selector is off
        }
    }

    let gates = [GateDescriptor.mul(colA: 0, colB: 1, colC: 2, selIdx: 0)]
    let result = engine.checkGateDescriptors(trace: [colA, colB, colC], gates: gates,
                                              constants: [], selectors: [sel],
                                              domainSize: domainSize)
    expect(result.isSatisfied, "gate descriptor with selector: wrong values masked out")
    print("  [OK] gate descriptor with selector")
}

// MARK: - Degree Analysis Tests

private func testDegreeAnalysis() {
    let engine = GPUConstraintSatEngine()

    let cs = ConstraintSystem(numWires: 3)
    // Linear: a - b = 0 (degree 1)
    cs.addConstraint(.wire(Wire.col(0)) - .wire(Wire.col(1)), label: "linear")
    // Quadratic: a*b - c = 0 (degree 2)
    cs.assertMul(Wire.col(0), Wire.col(1), Wire.col(2))
    // Boolean: a*(1-a) = 0 (degree 2)
    cs.assertBool(Wire.col(0))

    let degrees = engine.analyzeDegrees(cs)
    expectEqual(degrees.count, 3, "degree analysis: 3 constraints")
    expectEqual(degrees[0].degree, 1, "degree: linear is degree 1")
    expectEqual(degrees[1].degree, 2, "degree: mul is degree 2")
    expectEqual(degrees[2].degree, 2, "degree: bool is degree 2")
    print("  [OK] degree analysis")
}

private func testDegreeCheck() {
    let engine = GPUConstraintSatEngine()

    let cs = ConstraintSystem(numWires: 3)
    cs.addConstraint(.wire(Wire.col(0)) - .wire(Wire.col(1)), label: "linear")
    cs.assertMul(Wire.col(0), Wire.col(1), Wire.col(2))

    // Check max degree 1: mul constraint should violate
    let violations = engine.checkDegrees(cs, maxDegree: 1)
    expectEqual(violations.count, 1, "degree check: 1 violation for maxDegree=1")
    if !violations.isEmpty {
        expectEqual(violations[0].degree, 2, "degree check: violation is degree 2")
    }

    // Check max degree 2: no violations
    let noViolations = engine.checkDegrees(cs, maxDegree: 2)
    expectEqual(noViolations.count, 0, "degree check: no violations for maxDegree=2")
    print("  [OK] degree check")
}

private func testDegreePow() {
    let engine = GPUConstraintSatEngine()

    let cs = ConstraintSystem(numWires: 2)
    // a^3 - b = 0 (degree 3)
    cs.addConstraint(.pow(.wire(Wire.col(0)), 3) - .wire(Wire.col(1)), label: "cube")

    let degrees = engine.analyzeDegrees(cs)
    expectEqual(degrees.count, 1, "degree pow: 1 constraint")
    expectEqual(degrees[0].degree, 3, "degree pow: a^3 - b is degree 3")

    // Also check satisfaction
    let numRows = 4
    var colA = [Fr](), colB = [Fr]()
    for i in 0..<numRows {
        let a = frFromInt(UInt64(i + 1))
        colA.append(a)
        colB.append(frMul(a, frMul(a, a)))  // b = a^3
    }
    let result = engine.checkConstraintSystem(cs, trace: [colA, colB], numRows: numRows)
    expect(result.isSatisfied, "degree pow: a^3=b satisfied")
    print("  [OK] degree pow (a^3)")
}

// MARK: - Result and Violation Formatting Tests

private func testSatisfactionResultSummary() {
    let satisfied = SatisfactionResult(
        isSatisfied: true, firstViolation: nil,
        numConstraints: 10, numRows: 100,
        checkTimeMs: 1.5, usedGPU: true)
    expect(satisfied.summary.contains("SATISFIED"), "summary: contains SATISFIED")
    expect(satisfied.summary.contains("10"), "summary: contains constraint count")
    expect(satisfied.summary.contains("GPU"), "summary: contains GPU")

    let unsatisfied = SatisfactionResult(
        isSatisfied: false,
        firstViolation: ConstraintViolation(
            constraintIndex: 3, row: 7, residual: frFromInt(42),
            label: "test_gate", lhsValue: frFromInt(42), rhsValue: nil),
        numConstraints: 10, numRows: 100,
        checkTimeMs: 0.5, usedGPU: false)
    expect(unsatisfied.summary.contains("UNSATISFIED"), "summary: contains UNSATISFIED")
    print("  [OK] satisfaction result summary")
}

private func testViolationDescription() {
    let violation = ConstraintViolation(
        constraintIndex: 5, row: 12, residual: frFromInt(99),
        label: "my_gate", lhsValue: frFromInt(100), rhsValue: frFromInt(1))
    let desc = violation.description
    expect(desc.contains("constraint 5"), "violation desc: contains constraint index")
    expect(desc.contains("row 12"), "violation desc: contains row")
    expect(desc.contains("my_gate"), "violation desc: contains label")
    print("  [OK] violation description")
}

// MARK: - Public Input Injection Helper

private func testInjectPublicInputs() {
    let engine = GPUConstraintSatEngine()

    var z = [Fr.one, Fr.zero, Fr.zero, frFromInt(42)]
    let result = engine.injectPublicInputs(z: z, publicValues: [frFromInt(3), frFromInt(5)])
    expectEqual(result[0], Fr.one, "inject: z[0] unchanged (constant 1)")
    expectEqual(result[1], frFromInt(3), "inject: z[1] = public 0")
    expectEqual(result[2], frFromInt(5), "inject: z[2] = public 1")
    expectEqual(result[3], frFromInt(42), "inject: z[3] unchanged")
    print("  [OK] inject public inputs")
}

// MARK: - Large-Scale Tests

private func testLargeR1CS() {
    let engine = GPUConstraintSatEngine()
    let n = 1000
    let numVars = 3 * n + 1
    var aEntries = [(Int, Int, Fr)](), bEntries = [(Int, Int, Fr)](), cEntries = [(Int, Int, Fr)]()
    var z = [Fr](repeating: Fr.zero, count: numVars)
    z[0] = Fr.one
    for i in 0..<n {
        let (aIdx, bIdx, cIdx) = (1 + 3*i, 2 + 3*i, 3 + 3*i)
        aEntries.append((i, aIdx, Fr.one)); bEntries.append((i, bIdx, Fr.one)); cEntries.append((i, cIdx, Fr.one))
        let a = frFromInt(UInt64(i + 2)), b = frFromInt(UInt64(i + 3))
        z[aIdx] = a; z[bIdx] = b; z[cIdx] = frMul(a, b)
    }
    let A = buildSparseMatrix(rows: n, cols: numVars, entries: aEntries)
    let B = buildSparseMatrix(rows: n, cols: numVars, entries: bEntries)
    let C = buildSparseMatrix(rows: n, cols: numVars, entries: cEntries)
    let t0 = CFAbsoluteTimeGetCurrent()
    let result = engine.checkR1CS(R1CSSatInput(A: A, B: B, C: C, z: z, numPublicInputs: 0))
    let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
    expect(result.isSatisfied, "large R1CS: 1000 constraints satisfied")
    print("  [OK] large R1CS (1000 constraints, \(String(format: "%.2f", elapsed))ms)")
}

private func testLargePlonkish() {
    let engine = GPUConstraintSatEngine()
    let n = 500
    let negOne = frSub(Fr.zero, Fr.one)
    var gates = [PlonkGate](), wires = [[Int]](), witness = [Int: Fr]()
    for i in 0..<n {
        let (aVar, bVar, cVar) = (3*i, 3*i + 1, 3*i + 2)
        let a = frFromInt(UInt64(i + 1)), b = frFromInt(UInt64(i + 3))
        witness[aVar] = a; witness[bVar] = b
        if i % 2 == 0 {
            gates.append(PlonkGate(qL: Fr.one, qR: Fr.one, qO: negOne, qM: Fr.zero, qC: Fr.zero))
            witness[cVar] = frAdd(a, b)
        } else {
            gates.append(PlonkGate(qL: Fr.zero, qR: Fr.zero, qO: negOne, qM: Fr.one, qC: Fr.zero))
            witness[cVar] = frMul(a, b)
        }
        wires.append([aVar, bVar, cVar])
    }
    let t0 = CFAbsoluteTimeGetCurrent()
    let result = engine.checkPlonkish(PlonkishSatInput(gates: gates, wireAssignments: wires, witness: witness))
    let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
    expect(result.isSatisfied, "large Plonkish: 500 gates satisfied")
    print("  [OK] large Plonkish (500 gates, \(String(format: "%.2f", elapsed))ms)")
}

private func testLargeAIR() {
    let engine = GPUConstraintSatEngine()
    let numRows = 512
    var col0 = [Fr](), col1 = [Fr]()
    for i in 0..<numRows {
        let x = frFromInt(UInt64(i))
        col0.append(x); col1.append(frMul(x, x))
    }
    let t1: Expr = .wire(Wire(index: 0, row: 1)) - .wire(Wire.col(0)) - .constant(Fr.one)
    let t2: Expr = .wire(Wire.col(1)) - .mul(.wire(Wire.col(0)), .wire(Wire.col(0)))
    let t0 = CFAbsoluteTimeGetCurrent()
    let result = engine.checkAIR(AIRSatInput(trace: [col0, col1], transitionConstraints: [t1, t2]))
    let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
    expect(result.isSatisfied, "large AIR: 512 rows satisfied")
    print("  [OK] large AIR (512 rows, 2 constraints, \(String(format: "%.2f", elapsed))ms)")
}

// MARK: - Sparse Matrix Builder Helper

/// Build a SparseMatrix from a list of (row, col, value) entries.
private func buildSparseMatrix(rows: Int, cols: Int,
                                entries: [(Int, Int, Fr)]) -> SparseMatrix {
    var builder = SparseMatrixBuilder(rows: rows, cols: cols)
    for (r, c, v) in entries {
        builder.set(row: r, col: c, value: v)
    }
    return builder.build()
}
