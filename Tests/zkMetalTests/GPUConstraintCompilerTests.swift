// GPUConstraintCompilerEngine Tests — circuit to R1CS compilation correctness
//
// Tests the arithmetic circuit compiler for all supported gate types,
// verifying R1CS satisfaction, witness evaluation, and statistics.

import Foundation
import zkMetal

public func runGPUConstraintCompilerTests() {
    suite("GPUConstraintCompiler")

    testAddGateCompilation()
    testMulGateCompilation()
    testConstGateCompilation()
    testBoolConstraint()
    testBoolConstraintViolation()
    testRangeCheckConstraint()
    testRangeCheckViolation()
    testMuxConstraint()
    testMuxSelectZero()
    testMuxSelectOne()
    testCombinedCircuit()
    testWitnessEvaluation()
    testCompilationStats()
    testValidation()
    testValidationErrors()
    testToConstraintSystem()
    testToGateDescriptors()
    testBatchCompilation()
    testReset()
    testLargerCircuit()
    testChainedArithmetic()
    testPublicInputOrdering()
}

// MARK: - Addition Gate

private func testAddGateCompilation() {
    let compiler = GPUConstraintCompilerEngine()
    compiler.addPublicInput(wire: 0)
    compiler.addPublicInput(wire: 1)
    compiler.addGate(.add(left: 0, right: 1, out: 2))
    compiler.setWitness(wire: 0, value: frFromInt(3))
    compiler.setWitness(wire: 1, value: frFromInt(7))

    let result = compiler.compile()
    expect(result.isSatisfied(), "add gate: 3 + 7 = 10 should satisfy R1CS")
    expectEqual(result.stats.numConstraints, 1, "add gate: 1 constraint")
    expectEqual(result.stats.numAddGates, 1, "add gate: 1 add gate counted")
}

// MARK: - Multiplication Gate

private func testMulGateCompilation() {
    let compiler = GPUConstraintCompilerEngine()
    compiler.addPublicInput(wire: 0)
    compiler.addPublicInput(wire: 1)
    compiler.addGate(.mul(left: 0, right: 1, out: 2))
    compiler.setWitness(wire: 0, value: frFromInt(5))
    compiler.setWitness(wire: 1, value: frFromInt(6))

    let result = compiler.compile()
    expect(result.isSatisfied(), "mul gate: 5 * 6 = 30 should satisfy R1CS")
    expectEqual(result.stats.numMulGates, 1, "mul gate: 1 mul gate counted")
}

// MARK: - Constant Gate

private func testConstGateCompilation() {
    let compiler = GPUConstraintCompilerEngine()
    compiler.addGate(.const(out: 0, value: frFromInt(42)))
    compiler.addGate(.const(out: 1, value: frFromInt(13)))
    compiler.addGate(.add(left: 0, right: 1, out: 2))

    let result = compiler.compile()
    expect(result.isSatisfied(), "const gates: 42 + 13 = 55 should satisfy R1CS")
    expectEqual(result.stats.numConstGates, 2, "const gate: 2 const gates")
    expectEqual(result.stats.numConstraints, 3, "const gate: 3 constraints (2 const + 1 add)")
}

// MARK: - Boolean Constraint

private func testBoolConstraint() {
    let compiler = GPUConstraintCompilerEngine()
    compiler.addGate(.const(out: 0, value: Fr.one))
    compiler.addGate(.bool(wire: 0))
    compiler.setWitness(wire: 0, value: Fr.one)

    let result = compiler.compile()
    expect(result.isSatisfied(), "bool constraint: wire=1 should satisfy w*(1-w)=0")
}

private func testBoolConstraintViolation() {
    let compiler = GPUConstraintCompilerEngine()
    compiler.addGate(.bool(wire: 0))
    compiler.setWitness(wire: 0, value: frFromInt(2))

    let result = compiler.compile()
    expect(!result.isSatisfied(), "bool constraint: wire=2 should NOT satisfy w*(1-w)=0")
}

// MARK: - Range Check

private func testRangeCheckConstraint() {
    // Value 5 = 101 in binary (3 bits)
    let compiler = GPUConstraintCompilerEngine()
    compiler.addGate(.range(wire: 0, bits: 3, bitWires: [1, 2, 3]))
    compiler.setWitness(wire: 0, value: frFromInt(5))
    compiler.setWitness(wire: 1, value: Fr.one)   // bit 0 = 1
    compiler.setWitness(wire: 2, value: Fr.zero)   // bit 1 = 0
    compiler.setWitness(wire: 3, value: Fr.one)   // bit 2 = 1

    let result = compiler.compile()
    expect(result.isSatisfied(), "range check: 5 = 1*1 + 0*2 + 1*4 should satisfy")
    // 3 bool constraints + 1 decomposition = 4 constraints
    expectEqual(result.stats.numConstraints, 4, "range check: 4 constraints (3 bool + 1 decomp)")
}

private func testRangeCheckViolation() {
    // Value 5 but wrong decomposition
    let compiler = GPUConstraintCompilerEngine()
    compiler.addGate(.range(wire: 0, bits: 3, bitWires: [1, 2, 3]))
    compiler.setWitness(wire: 0, value: frFromInt(5))
    compiler.setWitness(wire: 1, value: Fr.one)   // bit 0 = 1
    compiler.setWitness(wire: 2, value: Fr.one)   // bit 1 = 1 (wrong!)
    compiler.setWitness(wire: 3, value: Fr.one)   // bit 2 = 1

    let result = compiler.compile()
    // 1 + 2 + 4 = 7 != 5, should not satisfy
    expect(!result.isSatisfied(), "range check: wrong decomposition should NOT satisfy")
}

// MARK: - Mux (Conditional Selection)

private func testMuxConstraint() {
    let compiler = GPUConstraintCompilerEngine()
    // sel=1: out = then_wire = 10
    compiler.addGate(.mux(sel: 0, thenWire: 1, elseWire: 2, out: 3))
    compiler.setWitness(wire: 0, value: Fr.one)      // sel = 1
    compiler.setWitness(wire: 1, value: frFromInt(10)) // then
    compiler.setWitness(wire: 2, value: frFromInt(20)) // else
    // out = else + sel*(then - else) = 20 + 1*(10-20) = 10
    compiler.setWitness(wire: 3, value: frFromInt(10))

    let result = compiler.compile()
    expect(result.isSatisfied(), "mux sel=1: out should be then_wire value")
}

private func testMuxSelectZero() {
    let compiler = GPUConstraintCompilerEngine()
    compiler.addGate(.mux(sel: 0, thenWire: 1, elseWire: 2, out: 3))
    compiler.setWitness(wire: 0, value: Fr.zero)      // sel = 0
    compiler.setWitness(wire: 1, value: frFromInt(10)) // then
    compiler.setWitness(wire: 2, value: frFromInt(20)) // else
    // out = else + 0*(then - else) = 20
    compiler.setWitness(wire: 3, value: frFromInt(20))

    let result = compiler.compile()
    expect(result.isSatisfied(), "mux sel=0: out should be else_wire value")
}

private func testMuxSelectOne() {
    let compiler = GPUConstraintCompilerEngine()
    compiler.addGate(.mux(sel: 0, thenWire: 1, elseWire: 2, out: 3))
    compiler.setWitness(wire: 0, value: Fr.one)       // sel = 1
    compiler.setWitness(wire: 1, value: frFromInt(99)) // then
    compiler.setWitness(wire: 2, value: frFromInt(42)) // else
    // out = 42 + 1*(99-42) = 99
    compiler.setWitness(wire: 3, value: frFromInt(99))

    let result = compiler.compile()
    expect(result.isSatisfied(), "mux sel=1: out should be 99")
}

// MARK: - Combined Circuit

private func testCombinedCircuit() {
    // Circuit: public inputs a, b
    // c = a * b
    // d = c + a
    // bool(e) where e = 1
    // result should satisfy
    let compiler = GPUConstraintCompilerEngine()
    compiler.addPublicInput(wire: 0)  // a
    compiler.addPublicInput(wire: 1)  // b
    compiler.addGate(.mul(left: 0, right: 1, out: 2))  // c = a * b
    compiler.addGate(.add(left: 2, right: 0, out: 3))  // d = c + a
    compiler.addGate(.const(out: 4, value: Fr.one))
    compiler.addGate(.bool(wire: 4))

    compiler.setWitness(wire: 0, value: frFromInt(3))
    compiler.setWitness(wire: 1, value: frFromInt(5))

    let result = compiler.compile()
    expect(result.isSatisfied(), "combined circuit: a=3, b=5, c=15, d=18, bool(1) should satisfy")
    expectEqual(result.stats.numConstraints, 4, "combined: 4 constraints")
}

// MARK: - Witness Evaluation

private func testWitnessEvaluation() {
    let compiler = GPUConstraintCompilerEngine()
    compiler.addGate(.const(out: 0, value: frFromInt(3)))
    compiler.addGate(.const(out: 1, value: frFromInt(7)))
    compiler.addGate(.mul(left: 0, right: 1, out: 2))
    compiler.addGate(.add(left: 2, right: 0, out: 3))

    let witness = compiler.evaluateWitness()
    expectEqual(witness[0]!, frFromInt(3), "witness eval: wire 0 = 3")
    expectEqual(witness[1]!, frFromInt(7), "witness eval: wire 1 = 7")
    expectEqual(witness[2]!, frFromInt(21), "witness eval: wire 2 = 3*7 = 21")
    expectEqual(witness[3]!, frFromInt(24), "witness eval: wire 3 = 21+3 = 24")
}

// MARK: - Compilation Statistics

private func testCompilationStats() {
    let compiler = GPUConstraintCompilerEngine()
    compiler.addPublicInput(wire: 0)
    compiler.addGate(.add(left: 0, right: 1, out: 2))
    compiler.addGate(.mul(left: 0, right: 1, out: 3))
    compiler.addGate(.const(out: 4, value: Fr.one))
    compiler.addGate(.bool(wire: 4))
    compiler.addGate(.range(wire: 5, bits: 2, bitWires: [6, 7]))
    compiler.addGate(.mux(sel: 4, thenWire: 2, elseWire: 3, out: 8))

    compiler.setWitness(wire: 0, value: frFromInt(1))
    compiler.setWitness(wire: 1, value: frFromInt(2))
    compiler.setWitness(wire: 5, value: frFromInt(3))
    compiler.setWitness(wire: 6, value: Fr.one)
    compiler.setWitness(wire: 7, value: Fr.one)

    let result = compiler.compile()
    let s = result.stats
    expectEqual(s.numAddGates, 1, "stats: 1 add gate")
    expectEqual(s.numMulGates, 1, "stats: 1 mul gate")
    expectEqual(s.numConstGates, 1, "stats: 1 const gate")
    expectEqual(s.numBoolConstraints, 1, "stats: 1 bool constraint")
    expectEqual(s.numRangeConstraints, 1, "stats: 1 range constraint")
    expectEqual(s.numMuxConstraints, 1, "stats: 1 mux constraint")
    expectEqual(s.numPublicInputs, 1, "stats: 1 public input")
    expect(s.compileTimeMs >= 0, "stats: compile time non-negative")
    expect(s.totalNnz > 0, "stats: non-zero NNZ")

    // Summary should be non-empty
    let summary = s.summary
    expect(summary.count > 0, "stats: summary is non-empty")
}

// MARK: - Validation

private func testValidation() {
    let compiler = GPUConstraintCompilerEngine()
    compiler.addPublicInput(wire: 0)
    compiler.addGate(.add(left: 0, right: 1, out: 2))
    compiler.addGate(.mul(left: 2, right: 1, out: 3))
    compiler.setWitness(wire: 0, value: frFromInt(1))
    compiler.setWitness(wire: 1, value: frFromInt(2))

    let errors = compiler.validate()
    expect(errors.isEmpty, "validation: valid circuit should have no errors")
}

private func testValidationErrors() {
    let compiler = GPUConstraintCompilerEngine()
    // Range with mismatched bit count
    compiler.addGate(.range(wire: 0, bits: 3, bitWires: [1, 2]))  // 2 wires for 3 bits
    let errors = compiler.validate()
    expect(!errors.isEmpty, "validation: mismatched bitWires count should error")
}

// MARK: - ConstraintSystem Conversion

private func testToConstraintSystem() {
    let compiler = GPUConstraintCompilerEngine()
    compiler.addGate(.mul(left: 0, right: 1, out: 2))
    compiler.addGate(.add(left: 2, right: 0, out: 3))
    compiler.addGate(.bool(wire: 4))

    let cs = compiler.toConstraintSystem()
    expectEqual(cs.constraints.count, 3, "toConstraintSystem: 3 constraints")
    expect(cs.numWires >= 5, "toConstraintSystem: at least 5 wires")
}

// MARK: - Gate Descriptor Conversion

private func testToGateDescriptors() {
    let compiler = GPUConstraintCompilerEngine()
    compiler.addGate(.add(left: 0, right: 1, out: 2))
    compiler.addGate(.mul(left: 0, right: 1, out: 3))
    compiler.addGate(.bool(wire: 4))

    let (descriptors, _) = compiler.toGateDescriptors()
    expectEqual(descriptors.count, 3, "toGateDescriptors: 3 descriptors")
    expectEqual(descriptors[0].type, ConstraintGateType.add.rawValue, "descriptor 0 is add")
    expectEqual(descriptors[1].type, ConstraintGateType.mul.rawValue, "descriptor 1 is mul")
    expectEqual(descriptors[2].type, ConstraintGateType.bool.rawValue, "descriptor 2 is bool")
}

// MARK: - Batch Compilation

private func testBatchCompilation() {
    // Two independent circuits merged into one R1CS
    let c1 = GPUConstraintCompilerEngine()
    c1.addPublicInput(wire: 0)
    c1.addGate(.mul(left: 0, right: 1, out: 2))
    c1.setWitness(wire: 0, value: frFromInt(3))
    c1.setWitness(wire: 1, value: frFromInt(4))

    let c2 = GPUConstraintCompilerEngine()
    c2.addPublicInput(wire: 0)
    c2.addGate(.add(left: 0, right: 1, out: 2))
    c2.setWitness(wire: 0, value: frFromInt(5))
    c2.setWitness(wire: 1, value: frFromInt(6))

    let result = GPUConstraintCompilerEngine.batchCompile([c1, c2])
    expect(result.isSatisfied(), "batch compile: both sub-circuits should satisfy")
    expectEqual(result.stats.numConstraints, 2, "batch: 2 constraints (1 per sub-circuit)")
    expectEqual(result.stats.numPublicInputs, 2, "batch: 2 public inputs")
}

// MARK: - Reset

private func testReset() {
    let compiler = GPUConstraintCompilerEngine()
    compiler.addPublicInput(wire: 0)
    compiler.addGate(.mul(left: 0, right: 1, out: 2))
    compiler.setWitness(wire: 0, value: frFromInt(1))

    compiler.reset()

    expectEqual(compiler.gateCount, 0, "reset: gate count is 0")
    expectEqual(compiler.wireCount, 0, "reset: wire count is 0")
    expectEqual(compiler.publicInputCount, 0, "reset: public input count is 0")
    expect(!compiler.hasWitness, "reset: no witness values")
}

// MARK: - Larger Circuit (stress test)

private func testLargerCircuit() {
    let compiler = GPUConstraintCompilerEngine()
    let n = 100  // chain of multiplications

    compiler.setWitness(wire: 0, value: frFromInt(2))
    compiler.setWitness(wire: 1, value: frFromInt(3))

    for i in 0..<n {
        let inA = i == 0 ? 0 : i + 1
        let inB = i == 0 ? 1 : i + 1
        let out = i + 2
        if i == 0 {
            compiler.addGate(.mul(left: 0, right: 1, out: 2))
        } else {
            // Each subsequent gate: out[i] = out[i-1] * 1
            // We use add for variety: out[i] = out[i-1] + 0
            compiler.addGate(.add(left: i + 1, right: i + 1, out: i + 2))
        }
    }

    let result = compiler.compile()
    expect(result.isSatisfied(), "larger circuit: 100 chained gates should satisfy")
    expectEqual(result.stats.numConstraints, n, "larger circuit: n constraints")
}

// MARK: - Chained Arithmetic

private func testChainedArithmetic() {
    // a=2, b=3, c=a*b=6, d=c+a=8, e=d*b=24
    let compiler = GPUConstraintCompilerEngine()
    compiler.addPublicInput(wire: 0)  // a
    compiler.addPublicInput(wire: 1)  // b
    compiler.addGate(.mul(left: 0, right: 1, out: 2))  // c = a*b
    compiler.addGate(.add(left: 2, right: 0, out: 3))  // d = c+a
    compiler.addGate(.mul(left: 3, right: 1, out: 4))  // e = d*b

    compiler.setWitness(wire: 0, value: frFromInt(2))
    compiler.setWitness(wire: 1, value: frFromInt(3))

    let witness = compiler.evaluateWitness()
    expectEqual(witness[2]!, frFromInt(6), "chained: c = 2*3 = 6")
    expectEqual(witness[3]!, frFromInt(8), "chained: d = 6+2 = 8")
    expectEqual(witness[4]!, frFromInt(24), "chained: e = 8*3 = 24")

    let result = compiler.compile()
    expect(result.isSatisfied(), "chained arithmetic should satisfy R1CS")
}

// MARK: - Public Input Ordering

private func testPublicInputOrdering() {
    // Verify public inputs appear in declared order in z vector
    let compiler = GPUConstraintCompilerEngine()
    compiler.addPublicInput(wire: 5)
    compiler.addPublicInput(wire: 2)
    compiler.addGate(.add(left: 5, right: 2, out: 10))
    compiler.setWitness(wire: 5, value: frFromInt(100))
    compiler.setWitness(wire: 2, value: frFromInt(200))

    let result = compiler.compile()
    // z[0] = 1, z[1] = public_0 (wire 5 = 100), z[2] = public_1 (wire 2 = 200)
    expectEqual(result.z[0], Fr.one, "z[0] = 1 (constant)")
    expectEqual(result.z[1], frFromInt(100), "z[1] = first public input (wire 5)")
    expectEqual(result.z[2], frFromInt(200), "z[2] = second public input (wire 2)")
    expect(result.isSatisfied(), "public input ordering: should satisfy")
}
