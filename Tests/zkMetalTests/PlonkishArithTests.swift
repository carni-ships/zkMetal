// PlonkishArithTests — Tests for the Plonkish arithmetization engine

import Foundation
import zkMetal

public func runPlonkishArithTests() {
    suite("PlonkishArith")

    testSimpleAddition()
    testMultiplicationCircuit()
    testMultiRegionLayout()
    testLookupArgument()
    testCustomGateUsage()
    testFullCompileRoundTrip()
    testCopyConstraints()
    testInstanceColumns()
}

// MARK: - Simple Arithmetic: a + b = c

private func testSimpleAddition() {
    let cs = PlonkishConstraintSystem()
    let a = cs.adviceColumn()   // advice 0
    let b = cs.adviceColumn()   // advice 1
    let c = cs.adviceColumn()   // advice 2
    let qL = cs.fixedColumn()   // fixed 0 = qL
    let qR = cs.fixedColumn()   // fixed 1 = qR
    let qO = cs.fixedColumn()   // fixed 2 = qO

    cs.enableEquality(a)
    cs.enableEquality(b)
    cs.enableEquality(c)

    // Gate: qL * a + qR * b + qO * c = 0
    cs.createGate(name: "add") { eval in
        let qLv = eval.queryFixed(column: 0)
        let qRv = eval.queryFixed(column: 1)
        let qOv = eval.queryFixed(column: 2)
        let av = eval.queryAdvice(column: 0)
        let bv = eval.queryAdvice(column: 1)
        let cv = eval.queryAdvice(column: 2)
        return frAdd(frAdd(frMul(qLv, av), frMul(qRv, bv)), frMul(qOv, cv))
    }

    let circuit = PlonkishCircuit(cs: cs)

    // Layout: a=3, b=5, c=8
    circuit.layoutRegion(name: "add") { region in
        region.assignAdvice(column: 0, offset: 0, value: frFromInt(3))
        region.assignAdvice(column: 1, offset: 0, value: frFromInt(5))
        region.assignAdvice(column: 2, offset: 0, value: frFromInt(8))
        // qL=1, qR=1, qO=-1: a + b - c = 0
        region.assignFixed(column: 0, offset: 0, value: Fr.one)
        region.assignFixed(column: 1, offset: 0, value: Fr.one)
        region.assignFixed(column: 2, offset: 0, value: frSub(Fr.zero, Fr.one))
    }

    let ok = circuit.verify()
    expect(ok, "a + b = c should verify (3 + 5 = 8)")

    // Negative test: wrong witness
    let circuit2 = PlonkishCircuit(cs: cs)
    circuit2.layoutRegion(name: "bad") { region in
        region.assignAdvice(column: 0, offset: 0, value: frFromInt(3))
        region.assignAdvice(column: 1, offset: 0, value: frFromInt(5))
        region.assignAdvice(column: 2, offset: 0, value: frFromInt(9)) // wrong!
        region.assignFixed(column: 0, offset: 0, value: Fr.one)
        region.assignFixed(column: 1, offset: 0, value: Fr.one)
        region.assignFixed(column: 2, offset: 0, value: frSub(Fr.zero, Fr.one))
    }
    let bad = circuit2.verify()
    expect(!bad, "a + b != c should fail verification (3 + 5 != 9)")
}

// MARK: - Multiplication with Witness Generation

private func testMultiplicationCircuit() {
    let cs = PlonkishConstraintSystem()
    let _ = cs.adviceColumn()  // advice 0 = a
    let _ = cs.adviceColumn()  // advice 1 = b
    let _ = cs.adviceColumn()  // advice 2 = c
    let _ = cs.fixedColumn()   // fixed 0 = qL
    let _ = cs.fixedColumn()   // fixed 1 = qR
    let _ = cs.fixedColumn()   // fixed 2 = qO
    let _ = cs.fixedColumn()   // fixed 3 = qM

    // Gate: qL*a + qR*b + qO*c + qM*a*b = 0
    cs.createGate(name: "arith") { eval in
        let qLv = eval.queryFixed(column: 0)
        let qRv = eval.queryFixed(column: 1)
        let qOv = eval.queryFixed(column: 2)
        let qMv = eval.queryFixed(column: 3)
        let av = eval.queryAdvice(column: 0)
        let bv = eval.queryAdvice(column: 1)
        let cv = eval.queryAdvice(column: 2)
        // qL*a + qR*b + qO*c + qM*a*b
        let linear = frAdd(frMul(qLv, av), frMul(qRv, bv))
        let output = frMul(qOv, cv)
        let mul = frMul(qMv, frMul(av, bv))
        return frAdd(frAdd(linear, output), mul)
    }

    let circuit = PlonkishCircuit(cs: cs)

    // Witness: a=7, b=11, c=77 with a*b=c
    circuit.layoutRegion(name: "mul") { region in
        let aVal = frFromInt(7)
        let bVal = frFromInt(11)
        let cVal = frMul(aVal, bVal) // 77

        region.assignAdvice(column: 0, offset: 0, value: aVal)
        region.assignAdvice(column: 1, offset: 0, value: bVal)
        region.assignAdvice(column: 2, offset: 0, value: cVal)
        // qM=1, qO=-1: a*b - c = 0
        region.assignFixed(column: 0, offset: 0, value: Fr.zero)
        region.assignFixed(column: 1, offset: 0, value: Fr.zero)
        region.assignFixed(column: 2, offset: 0, value: frSub(Fr.zero, Fr.one))
        region.assignFixed(column: 3, offset: 0, value: Fr.one)
    }

    let ok = circuit.verify()
    expect(ok, "a * b = c should verify (7 * 11 = 77)")
}

// MARK: - Multi-Region Layout

private func testMultiRegionLayout() {
    let cs = PlonkishConstraintSystem()
    let colA = cs.adviceColumn()  // 0
    let colB = cs.adviceColumn()  // 1
    let colC = cs.adviceColumn()  // 2
    let _ = cs.fixedColumn()      // 0 = qL
    let _ = cs.fixedColumn()      // 1 = qR
    let _ = cs.fixedColumn()      // 2 = qO
    let _ = cs.fixedColumn()      // 3 = qM

    cs.enableEquality(colA)
    cs.enableEquality(colC)

    cs.createGate(name: "arith") { eval in
        let qLv = eval.queryFixed(column: 0)
        let qRv = eval.queryFixed(column: 1)
        let qOv = eval.queryFixed(column: 2)
        let qMv = eval.queryFixed(column: 3)
        let av = eval.queryAdvice(column: 0)
        let bv = eval.queryAdvice(column: 1)
        let cv = eval.queryAdvice(column: 2)
        let linear = frAdd(frMul(qLv, av), frMul(qRv, bv))
        let output = frMul(qOv, cv)
        let mul = frMul(qMv, frMul(av, bv))
        return frAdd(frAdd(linear, output), mul)
    }

    let circuit = PlonkishCircuit(cs: cs)

    // Region 1: a * b = x (row 0)
    var xCell: PlonkishAssignedCell!
    circuit.layoutRegion(name: "mul") { region in
        let a = frFromInt(3)
        let b = frFromInt(4)
        let x = frMul(a, b) // 12
        region.assignAdvice(column: 0, offset: 0, value: a)
        region.assignAdvice(column: 1, offset: 0, value: b)
        xCell = region.assignAdvice(column: 2, offset: 0, value: x)
        region.assignFixed(column: 0, offset: 0, value: Fr.zero)
        region.assignFixed(column: 1, offset: 0, value: Fr.zero)
        region.assignFixed(column: 2, offset: 0, value: frSub(Fr.zero, Fr.one))
        region.assignFixed(column: 3, offset: 0, value: Fr.one)
    }

    // Region 2: x + y = z (row 1), with x wired from region 1
    circuit.layoutRegion(name: "add") { region in
        let xVal = frFromInt(12)
        let y = frFromInt(5)
        let z = frAdd(xVal, y) // 17
        let xHere = region.assignAdvice(column: 0, offset: 0, value: xVal)
        region.assignAdvice(column: 1, offset: 0, value: y)
        region.assignAdvice(column: 2, offset: 0, value: z)
        region.assignFixed(column: 0, offset: 0, value: Fr.one)
        region.assignFixed(column: 1, offset: 0, value: Fr.one)
        region.assignFixed(column: 2, offset: 0, value: frSub(Fr.zero, Fr.one))
        region.assignFixed(column: 3, offset: 0, value: Fr.zero)
        // Copy constraint: x from region 1 = a in region 2
        region.constrainEqual(xCell, xHere)
    }

    let ok = circuit.verify()
    expect(ok, "Multi-region: (3*4) + 5 = 17")
    expect(circuit.usedRows == 2, "Used 2 rows across 2 regions")
}

// MARK: - Lookup Argument

private func testLookupArgument() {
    let cs = PlonkishConstraintSystem()
    let _ = cs.adviceColumn()  // 0 = input
    let _ = cs.adviceColumn()  // 1 = table
    let _ = cs.fixedColumn()   // 0 = lookup enable

    // Lookup: when fixed[0]=1, advice[0] must be in advice[1]
    cs.createLookup(name: "range_check",
        inputExpr: { eval in eval.queryAdvice(column: 0) },
        tableExpr: { eval in eval.queryAdvice(column: 1) })

    let circuit = PlonkishCircuit(cs: cs)

    // Register a small lookup table
    let tableValues: [Fr] = (0..<8).map { frFromInt(UInt64($0)) }
    circuit.addLookupTable(values: tableValues)

    // Layout: assign input values and table
    circuit.layoutRegion(name: "lookups", numRows: 4) { region in
        // Input values (all in range 0..7)
        region.assignAdvice(column: 0, offset: 0, value: frFromInt(3))
        region.assignAdvice(column: 0, offset: 1, value: frFromInt(7))
        region.assignAdvice(column: 0, offset: 2, value: frFromInt(0))
        region.assignAdvice(column: 0, offset: 3, value: frFromInt(5))

        // Table column values
        for i in 0..<4 {
            region.assignAdvice(column: 1, offset: i, value: frFromInt(UInt64(i)))
        }
    }

    // Verify the circuit compiles
    let result = circuit.compile()
    expect(result.circuit.numGates >= 4, "Lookup circuit has gates")
    expect(result.circuit.lookupTables.count == 1, "One lookup table registered")
    expect(result.circuit.lookupTables[0].values.count == 8, "Table has 8 entries")
}

// MARK: - Custom Gate Usage

private func testCustomGateUsage() {
    let cs = PlonkishConstraintSystem()
    let colA = cs.adviceColumn()  // 0
    let _ = cs.adviceColumn()     // 1
    let _ = cs.adviceColumn()     // 2
    let _ = cs.fixedColumn()      // 0 = qBool

    // Bool gate: qBool * a * (1 - a) = 0
    cs.createGate(name: "bool_check") { eval in
        let qBool = eval.queryFixed(column: 0)
        let a = eval.queryAdvice(column: 0)
        // qBool * a * (1 - a)
        let oneMinusA = frSub(Fr.one, a)
        return frMul(qBool, frMul(a, oneMinusA))
    }

    // Also register the BoolGate template
    cs.addCustomGate(BoolGate(), columns: [colA])

    let circuit = PlonkishCircuit(cs: cs)

    // Row 0: a=0 (boolean, OK)
    // Row 1: a=1 (boolean, OK)
    circuit.layoutRegion(name: "bools", numRows: 2) { region in
        region.assignAdvice(column: 0, offset: 0, value: Fr.zero)
        region.assignFixed(column: 0, offset: 0, value: Fr.one) // enable bool check
        region.assignAdvice(column: 0, offset: 1, value: Fr.one)
        region.assignFixed(column: 0, offset: 1, value: Fr.one)
    }

    let ok = circuit.verify()
    expect(ok, "Bool gate: 0 and 1 are valid booleans")

    // Negative: a=2 is not boolean
    let circuit2 = PlonkishCircuit(cs: cs)
    circuit2.layoutRegion(name: "bad_bool") { region in
        region.assignAdvice(column: 0, offset: 0, value: frFromInt(2))
        region.assignFixed(column: 0, offset: 0, value: Fr.one)
    }
    let bad = circuit2.verify()
    expect(!bad, "Bool gate: 2 is not a valid boolean")

    // Verify custom gate templates are tracked
    expect(cs.customGateTemplates.count == 1, "One custom gate template registered")
}

// MARK: - Full Compile -> PlonkCircuit Round-trip

private func testFullCompileRoundTrip() {
    let cs = PlonkishConstraintSystem()
    let _ = cs.adviceColumn()  // 0 = a
    let _ = cs.adviceColumn()  // 1 = b
    let _ = cs.adviceColumn()  // 2 = c
    let _ = cs.fixedColumn()   // 0 = qL
    let _ = cs.fixedColumn()   // 1 = qR
    let _ = cs.fixedColumn()   // 2 = qO
    let _ = cs.fixedColumn()   // 3 = qM
    let _ = cs.fixedColumn()   // 4 = qC
    let _ = cs.instanceColumn() // 0 = public output

    cs.createGate(name: "arith") { eval in
        let qL = eval.queryFixed(column: 0)
        let qR = eval.queryFixed(column: 1)
        let qO = eval.queryFixed(column: 2)
        let qM = eval.queryFixed(column: 3)
        let qC = eval.queryFixed(column: 4)
        let a = eval.queryAdvice(column: 0)
        let b = eval.queryAdvice(column: 1)
        let c = eval.queryAdvice(column: 2)
        let linear = frAdd(frMul(qL, a), frMul(qR, b))
        let output = frMul(qO, c)
        let mul = frMul(qM, frMul(a, b))
        return frAdd(frAdd(frAdd(linear, output), mul), qC)
    }

    let circuit = PlonkishCircuit(cs: cs)

    // Row 0: a*b - c = 0 (3 * 5 = 15)
    // Row 1: c + 2 - d = 0 (15 + 2 = 17)
    circuit.layoutRegion(name: "compute", numRows: 2) { region in
        let a = frFromInt(3)
        let b = frFromInt(5)
        let c = frMul(a, b) // 15
        let d = frAdd(c, frFromInt(2)) // 17

        // Row 0: mul gate
        region.assignAdvice(column: 0, offset: 0, value: a)
        region.assignAdvice(column: 1, offset: 0, value: b)
        region.assignAdvice(column: 2, offset: 0, value: c)
        region.assignFixed(column: 0, offset: 0, value: Fr.zero) // qL
        region.assignFixed(column: 1, offset: 0, value: Fr.zero) // qR
        region.assignFixed(column: 2, offset: 0, value: frSub(Fr.zero, Fr.one)) // qO
        region.assignFixed(column: 3, offset: 0, value: Fr.one)  // qM
        region.assignFixed(column: 4, offset: 0, value: Fr.zero) // qC

        // Row 1: add-const gate: qL*a + qC + qO*c = 0 => a + 2 - c = 0
        region.assignAdvice(column: 0, offset: 1, value: c) // input = 15
        region.assignAdvice(column: 1, offset: 1, value: Fr.zero) // unused
        region.assignAdvice(column: 2, offset: 1, value: d) // output = 17
        region.assignFixed(column: 0, offset: 1, value: Fr.one)  // qL
        region.assignFixed(column: 1, offset: 1, value: Fr.zero) // qR
        region.assignFixed(column: 2, offset: 1, value: frSub(Fr.zero, Fr.one)) // qO
        region.assignFixed(column: 3, offset: 1, value: Fr.zero) // qM
        region.assignFixed(column: 4, offset: 1, value: frFromInt(2)) // qC
    }

    let ok = circuit.verify()
    expect(ok, "Full circuit: 3*5=15, 15+2=17 verifies")

    // Compile to PlonkCircuit
    let result = circuit.compile()
    expect(result.circuit.numGates >= 2, "Compiled circuit has >= 2 gates")
    expect(result.circuit.numGates & (result.circuit.numGates - 1) == 0,
           "Gate count is power of 2 (padded)")
    expect(result.witnessPolynomials.count == 3, "3 witness polynomials (advice cols)")
    expect(result.numRows >= 4, "At least 4 rows (power of 2)")

    // Verify witness polynomials contain correct values
    let wA = result.witnessPolynomials[0]
    let wB = result.witnessPolynomials[1]
    let wC = result.witnessPolynomials[2]
    expect(frEqual(wA[0], frFromInt(3)), "Witness a[0] = 3")
    expect(frEqual(wB[0], frFromInt(5)), "Witness b[0] = 5")
    expect(frEqual(wC[0], frFromInt(15)), "Witness c[0] = 15")
    expect(frEqual(wA[1], frFromInt(15)), "Witness a[1] = 15")
    expect(frEqual(wC[1], frFromInt(17)), "Witness c[1] = 17")

    // Verify the PlonkCircuit gate structure
    let g0 = result.circuit.gates[0]
    expect(frEqual(g0.qM, Fr.one), "Gate 0: qM=1 (mul gate)")
    let g1 = result.circuit.gates[1]
    expect(frEqual(g1.qL, Fr.one), "Gate 1: qL=1 (add gate)")
    expect(frEqual(g1.qC, frFromInt(2)), "Gate 1: qC=2 (constant)")
}

// MARK: - Copy Constraints

private func testCopyConstraints() {
    let cs = PlonkishConstraintSystem()
    let colA = cs.adviceColumn()  // 0
    let colB = cs.adviceColumn()  // 1
    cs.enableEquality(colA)
    cs.enableEquality(colB)

    // No gates - just test copy constraints
    let circuit = PlonkishCircuit(cs: cs)

    var cell1: PlonkishAssignedCell!
    var cell2: PlonkishAssignedCell!

    circuit.layoutRegion(name: "r1") { region in
        cell1 = region.assignAdvice(column: 0, offset: 0, value: frFromInt(42))
    }
    circuit.layoutRegion(name: "r2") { region in
        cell2 = region.assignAdvice(column: 1, offset: 0, value: frFromInt(42))
        region.constrainEqual(cell1, cell2)
    }

    let ok = circuit.verify()
    expect(ok, "Copy constraint: same value passes")

    // Test with mismatched values
    let circuit2 = PlonkishCircuit(cs: cs)
    circuit2.layoutRegion(name: "r1") { region in
        cell1 = region.assignAdvice(column: 0, offset: 0, value: frFromInt(42))
    }
    circuit2.layoutRegion(name: "r2") { region in
        cell2 = region.assignAdvice(column: 1, offset: 0, value: frFromInt(99))
        region.constrainEqual(cell1, cell2)
    }

    let bad = circuit2.verify()
    expect(!bad, "Copy constraint: different values fails")
}

// MARK: - Instance Columns

private func testInstanceColumns() {
    let cs = PlonkishConstraintSystem()
    let colA = cs.adviceColumn()
    let colI = cs.instanceColumn()
    cs.enableEquality(colA)
    cs.enableEquality(colI)

    let circuit = PlonkishCircuit(cs: cs)
    circuit.layoutRegion(name: "pub") { region in
        let adv = region.assignAdvice(column: 0, offset: 0, value: frFromInt(100))
        region.assignInstance(column: 0, offset: 0, value: frFromInt(100), constrainTo: adv)
    }

    let ok = circuit.verify()
    expect(ok, "Instance column matches advice via copy constraint")

    // Compile and check public inputs
    let result = circuit.compile()
    expect(!result.circuit.publicInputIndices.isEmpty, "Has public input indices")
    expect(result.witness.instance.count == 1, "One instance column")
    expect(frEqual(result.witness.instance[0][0], frFromInt(100)), "Instance value = 100")
}
