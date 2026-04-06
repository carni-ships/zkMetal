// GPU R1CS-to-AIR Compiler Engine tests
import zkMetal
import Foundation

public func runGPUR1CSToAIRCompilerTests() {
    // Wire access analysis
    suite("R1CS-to-AIR: wire access analysis")
    testWireAccessBasic()
    testWireAccessFrequencySorting()
    testWireAccessCoOccurrence()
    testWireAccessDensity()

    // Uniformity analysis
    suite("R1CS-to-AIR: uniformity analysis")
    testUniformSingleConstraintType()
    testNonUniformMixedConstraints()
    testUniformityEmptyR1CS()

    // Trace layout
    suite("R1CS-to-AIR: trace layout")
    testLayoutColumnAssignment()
    testLayoutPowerOfTwoPadding()
    testLayoutSelectorColumns()
    testLayoutWireReordering()

    // Transition constraints
    suite("R1CS-to-AIR: transition constraints")
    testTransitionFromMultiplication()
    testTransitionFromAddition()
    testTransitionMultipleConstraints()
    testTransitionUniformMerge()

    // Boundary constraints
    suite("R1CS-to-AIR: boundary constraints")
    testBoundaryConstantOne()
    testBoundaryPublicInputs()
    testBoundaryMultiplePublicInputs()

    // CPU trace generation
    suite("R1CS-to-AIR: CPU trace generation")
    testCPUTraceMultiplicationGate()
    testCPUTraceAdditionGate()
    testCPUTraceMultiConstraint()
    testCPUTracePaddingZeros()

    // Full compilation pipeline
    suite("R1CS-to-AIR: full compilation")
    testCompileSimpleMultiplication()
    testCompileAdditionCircuit()
    testCompileMultiConstraintCircuit()
    testCompileNonUniformCircuit()

    // Trace verification
    suite("R1CS-to-AIR: trace verification")
    testVerifyValidTrace()
    testVerifyInvalidTrace()
    testVerifyBoundaryViolation()

    // Batch constraint evaluation
    suite("R1CS-to-AIR: batch constraint evaluation")
    testBatchEvalSatisfied()
    testBatchEvalUnsatisfied()
    testBatchEvalWithResiduals()

    // FrAIRExpression conversion
    suite("R1CS-to-AIR: expression conversion")
    testExpressionSingleTerm()
    testExpressionMultiTerm()
    testExpressionMultiConstraint()

    // Density analysis
    suite("R1CS-to-AIR: density stats")
    testDensitySparse()
    testDensityDense()

    // Vanishing polynomial
    suite("R1CS-to-AIR: vanishing polynomial")
    testVanishingAtRootOfUnity()
    testVanishingAtRandomPoint()

    // Constraint composition
    suite("R1CS-to-AIR: constraint composition")
    testR1CSAIRCompositionSingleConstraint()
    testR1CSAIRCompositionMultipleConstraints()

    // Selector polynomials
    suite("R1CS-to-AIR: selector polynomials")
    testSelectorUniform()
    testSelectorNonUniform()

    // Quick verify
    suite("R1CS-to-AIR: quick verify")
    testQuickVerifyValid()
    testQuickVerifyInvalid()

    // Public input extraction
    suite("R1CS-to-AIR: public input extraction")
    testExtractPublicInputs()

    // Trace column reorder
    suite("R1CS-to-AIR: column reorder")
    testColumnReorderIdentity()
    testColumnReorderSwap()

    // Degree analysis
    suite("R1CS-to-AIR: degree analysis")
    testDegreeAnalysis()

    // Error handling
    suite("R1CS-to-AIR: error handling")
    testInsufficientWitness()

    // GPU trace generation (falls back to CPU if no GPU)
    suite("R1CS-to-AIR: GPU trace generation")
    testGPUTraceMultiplicationGate()
    testGPUTraceMultiConstraint()
}

// MARK: - Helper: build simple R1CS

/// Build a single multiplication gate: a * b = c
/// Wires: [1, a, b, c], 1 constraint, 1 public input
private func makeMultiplicationR1CS() -> (R1CSInstance, [Fr]) {
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    // Constraint 0: w[1] * w[2] = w[3]
    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    bE.append(R1CSEntry(row: 0, col: 2, val: one))
    cE.append(R1CSEntry(row: 0, col: 3, val: one))

    let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 1,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    let a = frFromInt(3)
    let b = frFromInt(7)
    let c = frMul(a, b) // 21
    let witness = [Fr.one, a, b, c]

    return (r1cs, witness)
}

/// Build an addition gate encoded as R1CS: (a + b) * 1 = c
/// Wires: [1, a, b, c], 1 constraint, 1 public input
private func makeAdditionR1CS() -> (R1CSInstance, [Fr]) {
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    // (w[1] + w[2]) * 1 = w[3]
    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    aE.append(R1CSEntry(row: 0, col: 2, val: one))
    bE.append(R1CSEntry(row: 0, col: 0, val: one))  // * 1 (constant wire)
    cE.append(R1CSEntry(row: 0, col: 3, val: one))

    let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 1,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    let a = frFromInt(5)
    let b = frFromInt(12)
    let c = frAdd(a, b) // 17
    let witness = [Fr.one, a, b, c]

    return (r1cs, witness)
}

/// Build a multi-constraint circuit:
/// Constraint 0: w[1] * w[2] = w[3]  (a*b = t1)
/// Constraint 1: w[3] * w[4] = w[5]  (t1*c = t2)
/// Wires: [1, a, b, t1, c, t2], 2 constraints, 2 public inputs
private func makeMultiConstraintR1CS() -> (R1CSInstance, [Fr]) {
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    // Constraint 0: w[1] * w[2] = w[3]
    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    bE.append(R1CSEntry(row: 0, col: 2, val: one))
    cE.append(R1CSEntry(row: 0, col: 3, val: one))

    // Constraint 1: w[3] * w[4] = w[5]
    aE.append(R1CSEntry(row: 1, col: 3, val: one))
    bE.append(R1CSEntry(row: 1, col: 4, val: one))
    cE.append(R1CSEntry(row: 1, col: 5, val: one))

    let r1cs = R1CSInstance(numConstraints: 2, numVars: 6, numPublic: 2,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    let a = frFromInt(3)
    let b = frFromInt(5)
    let t1 = frMul(a, b) // 15
    let c = frFromInt(2)
    let t2 = frMul(t1, c) // 30
    let witness = [Fr.one, a, b, t1, c, t2]

    return (r1cs, witness)
}

/// Build a non-uniform circuit with different constraint structures:
/// Constraint 0: w[1] * w[2] = w[3]     (2 terms in A: 1*w[1])
/// Constraint 1: (w[1]+w[2]) * w[0] = w[3]  (2 terms in A: w[1]+w[2])
/// These have different A-sparsity patterns.
private func makeNonUniformR1CS() -> (R1CSInstance, [Fr]) {
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    // Constraint 0: w[1] * w[2] = w[3]
    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    bE.append(R1CSEntry(row: 0, col: 2, val: one))
    cE.append(R1CSEntry(row: 0, col: 3, val: one))

    // Constraint 1: (w[1] + w[2]) * 1 = w[4]
    aE.append(R1CSEntry(row: 1, col: 1, val: one))
    aE.append(R1CSEntry(row: 1, col: 2, val: one))
    bE.append(R1CSEntry(row: 1, col: 0, val: one))
    cE.append(R1CSEntry(row: 1, col: 4, val: one))

    let r1cs = R1CSInstance(numConstraints: 2, numVars: 5, numPublic: 1,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    let a = frFromInt(4)
    let b = frFromInt(6)
    let t1 = frMul(a, b) // 24
    let t2 = frAdd(a, b) // 10
    let witness = [Fr.one, a, b, t1, t2]

    return (r1cs, witness)
}

// MARK: - Wire Access Analysis Tests

private func testWireAccessBasic() {
    let (r1cs, _) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let analysis = engine.analyzeWireAccess(r1cs: r1cs)

    expectEqual(analysis.accessCounts.count, 4, "4 wires in multiplication circuit")
    // Wire 0 (constant): not referenced in this circuit's A,B,C entries
    expectEqual(analysis.accessCounts[0], 0, "wire 0 not referenced")
    // Wire 1 (a): referenced once in A
    expectEqual(analysis.accessCounts[1], 1, "wire 1 referenced once")
    // Wire 2 (b): referenced once in B
    expectEqual(analysis.accessCounts[2], 1, "wire 2 referenced once")
    // Wire 3 (c): referenced once in C
    expectEqual(analysis.accessCounts[3], 1, "wire 3 referenced once")
}

private func testWireAccessFrequencySorting() {
    let (r1cs, _) = makeMultiConstraintR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let analysis = engine.analyzeWireAccess(r1cs: r1cs)

    // Wire 3 (t1) appears in both constraints: A row 1 + C row 0 = 2 references
    expectEqual(analysis.accessCounts[3], 2, "wire 3 (t1) referenced twice")
    // Wire 3 should be first in sorted order (highest frequency)
    expectEqual(analysis.sortedByFrequency[0], 3, "most-accessed wire is t1 (wire 3)")
}

private func testWireAccessCoOccurrence() {
    let (r1cs, _) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let analysis = engine.analyzeWireAccess(r1cs: r1cs)

    // Constraint 0 uses wires {1, 2, 3} => C(3,2) = 3 co-occurrences
    expectEqual(analysis.coOccurrenceCount, 3, "3 co-occurrences in single constraint")
}

private func testWireAccessDensity() {
    let (r1cs, _) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let analysis = engine.analyzeWireAccess(r1cs: r1cs)

    // 3 total entries / 1 constraint = 3.0
    expect(analysis.avgDensity > 2.9 && analysis.avgDensity < 3.1,
           "average density ~3.0 for simple mult gate")
}

// MARK: - Uniformity Analysis Tests

private func testUniformSingleConstraintType() {
    let (r1cs, _) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let analysis = engine.analyzeUniformity(r1cs: r1cs)

    expect(analysis.isUniform, "single constraint is uniform")
    expectEqual(analysis.numDistinctStructures, 1, "1 distinct structure")
}

private func testNonUniformMixedConstraints() {
    let (r1cs, _) = makeNonUniformR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let analysis = engine.analyzeUniformity(r1cs: r1cs)

    expect(!analysis.isUniform, "mixed constraint types are non-uniform")
    expectEqual(analysis.numDistinctStructures, 2, "2 distinct structures")
    // First constraint: (1A, 1B, 1C), second: (2A, 1B, 1C)
    expectEqual(analysis.constraintStructureMap.count, 2, "2 constraints mapped")
    expect(analysis.constraintStructureMap[0] != analysis.constraintStructureMap[1],
           "different constraints map to different structures")
}

private func testUniformityEmptyR1CS() {
    let r1cs = R1CSInstance(numConstraints: 0, numVars: 1, numPublic: 0,
                            aEntries: [], bEntries: [], cEntries: [])
    let engine = GPUR1CSToAIRCompilerEngine()
    let analysis = engine.analyzeUniformity(r1cs: r1cs)

    expect(analysis.isUniform, "empty R1CS is trivially uniform")
    expectEqual(analysis.numDistinctStructures, 0, "0 structures")
}

// MARK: - Trace Layout Tests

private func testLayoutColumnAssignment() {
    let (r1cs, _) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let wireAnalysis = engine.analyzeWireAccess(r1cs: r1cs)
    let uniformity = engine.analyzeUniformity(r1cs: r1cs)
    let layout = engine.computeLayout(r1cs: r1cs, wireAnalysis: wireAnalysis,
                                      uniformityAnalysis: uniformity)

    expectEqual(layout.aValColumn, 0, "A values in column 0")
    expectEqual(layout.bValColumn, 1, "B values in column 1")
    expectEqual(layout.cValColumn, 2, "C values in column 2")
    expectEqual(layout.wireColumnOffset, 3, "wire columns start at 3")
    expectEqual(layout.numWireColumns, 4, "4 wire columns")
    // Total: 3 (A,B,C) + 4 (wires) + 0 (selectors for uniform) = 7
    expectEqual(layout.numColumns, 7, "7 total columns")
}

private func testLayoutPowerOfTwoPadding() {
    let (r1cs, _) = makeMultiConstraintR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let wireAnalysis = engine.analyzeWireAccess(r1cs: r1cs)
    let uniformity = engine.analyzeUniformity(r1cs: r1cs)
    let layout = engine.computeLayout(r1cs: r1cs, wireAnalysis: wireAnalysis,
                                      uniformityAnalysis: uniformity)

    // 2 constraints -> pad to 2 (already power of 2)
    expectEqual(layout.numRows, 2, "2 constraints pad to 2 rows")
    expectEqual(layout.logNumRows, 1, "log2(2) = 1")
}

private func testLayoutSelectorColumns() {
    let (r1cs, _) = makeNonUniformR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let wireAnalysis = engine.analyzeWireAccess(r1cs: r1cs)
    let uniformity = engine.analyzeUniformity(r1cs: r1cs)
    let layout = engine.computeLayout(r1cs: r1cs, wireAnalysis: wireAnalysis,
                                      uniformityAnalysis: uniformity)

    expectEqual(layout.numSelectorColumns, 2, "2 selector columns for non-uniform")
    expectEqual(layout.selectorColumnOffset, layout.wireColumnOffset + layout.numWireColumns,
                "selectors after wire columns")
}

private func testLayoutWireReordering() {
    let (r1cs, _) = makeMultiConstraintR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    engine.enableLayoutOptimization = true
    let wireAnalysis = engine.analyzeWireAccess(r1cs: r1cs)
    let uniformity = engine.analyzeUniformity(r1cs: r1cs)
    let layout = engine.computeLayout(r1cs: r1cs, wireAnalysis: wireAnalysis,
                                      uniformityAnalysis: uniformity)

    // Wire 3 is most-accessed (2 refs), should be first in column mapping
    let wire3col = layout.wireToColumn[3]
    expectEqual(wire3col, layout.wireColumnOffset, "wire 3 maps to first wire column")
}

// MARK: - Transition Constraint Tests

private func testTransitionFromMultiplication() {
    let (r1cs, _) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let uniformity = engine.analyzeUniformity(r1cs: r1cs)
    let constraints = engine.buildTransitionConstraints(r1cs: r1cs,
                                                         uniformityAnalysis: uniformity)

    expectEqual(constraints.count, 1, "uniform R1CS: 1 transition constraint")
    expectEqual(constraints[0].degree, 2, "R1CS constraint has degree 2")
    expectEqual(constraints[0].aTerms.count, 1, "A has 1 term")
    expectEqual(constraints[0].bTerms.count, 1, "B has 1 term")
    expectEqual(constraints[0].cTerms.count, 1, "C has 1 term")
}

private func testTransitionFromAddition() {
    let (r1cs, _) = makeAdditionR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let uniformity = engine.analyzeUniformity(r1cs: r1cs)
    let constraints = engine.buildTransitionConstraints(r1cs: r1cs,
                                                         uniformityAnalysis: uniformity)

    expectEqual(constraints.count, 1, "1 transition constraint")
    // A has 2 terms: w[1] + w[2]
    expectEqual(constraints[0].aTerms.count, 2, "A has 2 terms for addition gate")
    // B has 1 term: w[0] (= 1)
    expectEqual(constraints[0].bTerms.count, 1, "B has 1 term (constant)")
}

private func testTransitionMultipleConstraints() {
    let (r1cs, _) = makeMultiConstraintR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let uniformity = engine.analyzeUniformity(r1cs: r1cs)
    let constraints = engine.buildTransitionConstraints(r1cs: r1cs,
                                                         uniformityAnalysis: uniformity)

    // Multi-constraint with same structure (1A,1B,1C) => uniform, 1 constraint
    expectEqual(constraints.count, 1,
                "uniform multi-constraint: 1 template constraint")
}

private func testTransitionUniformMerge() {
    let (r1cs, _) = makeNonUniformR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let uniformity = engine.analyzeUniformity(r1cs: r1cs)
    let constraints = engine.buildTransitionConstraints(r1cs: r1cs,
                                                         uniformityAnalysis: uniformity)

    // Non-uniform: 2 constraints, each with its own structure
    expectEqual(constraints.count, 2, "non-uniform: 2 transition constraints")
    expect(constraints[0].selectorIndex != nil, "non-uniform constraints have selectors")
}

// MARK: - Boundary Constraint Tests

private func testBoundaryConstantOne() {
    let (r1cs, witness) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let wireAnalysis = engine.analyzeWireAccess(r1cs: r1cs)
    let uniformity = engine.analyzeUniformity(r1cs: r1cs)
    let layout = engine.computeLayout(r1cs: r1cs, wireAnalysis: wireAnalysis,
                                      uniformityAnalysis: uniformity)
    let boundaries = engine.buildBoundaryConstraints(r1cs: r1cs, witness: witness,
                                                     layout: layout)

    // Should have constant-one boundary + 1 public input = 2 boundaries
    expect(boundaries.count >= 1, "at least 1 boundary (constant one wire)")
    let oneBC = boundaries.first { $0.label == "constant_one_wire" }
    expect(oneBC != nil, "constant one wire boundary exists")
    if let bc = oneBC {
        expect(frEqual(bc.value, Fr.one), "constant one wire value is 1")
    }
}

private func testBoundaryPublicInputs() {
    let (r1cs, witness) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let wireAnalysis = engine.analyzeWireAccess(r1cs: r1cs)
    let uniformity = engine.analyzeUniformity(r1cs: r1cs)
    let layout = engine.computeLayout(r1cs: r1cs, wireAnalysis: wireAnalysis,
                                      uniformityAnalysis: uniformity)
    let boundaries = engine.buildBoundaryConstraints(r1cs: r1cs, witness: witness,
                                                     layout: layout)

    // 1 public input => "public_input_1"
    let pubBC = boundaries.first { $0.label == "public_input_1" }
    expect(pubBC != nil, "public input 1 boundary exists")
    if let bc = pubBC {
        expect(frEqual(bc.value, witness[1]), "public input value matches witness[1]")
    }
}

private func testBoundaryMultiplePublicInputs() {
    let (r1cs, witness) = makeMultiConstraintR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let wireAnalysis = engine.analyzeWireAccess(r1cs: r1cs)
    let uniformity = engine.analyzeUniformity(r1cs: r1cs)
    let layout = engine.computeLayout(r1cs: r1cs, wireAnalysis: wireAnalysis,
                                      uniformityAnalysis: uniformity)
    let boundaries = engine.buildBoundaryConstraints(r1cs: r1cs, witness: witness,
                                                     layout: layout)

    // 2 public inputs + 1 constant = 3 boundaries
    expectEqual(boundaries.count, 3, "3 boundary constraints (1 + 2 public)")
    let pub1 = boundaries.first { $0.label == "public_input_1" }
    let pub2 = boundaries.first { $0.label == "public_input_2" }
    expect(pub1 != nil, "public_input_1 exists")
    expect(pub2 != nil, "public_input_2 exists")
}

// MARK: - CPU Trace Generation Tests

private func testCPUTraceMultiplicationGate() {
    let (r1cs, witness) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let wireAnalysis = engine.analyzeWireAccess(r1cs: r1cs)
    let uniformity = engine.analyzeUniformity(r1cs: r1cs)
    let layout = engine.computeLayout(r1cs: r1cs, wireAnalysis: wireAnalysis,
                                      uniformityAnalysis: uniformity)
    let constraints = engine.buildTransitionConstraints(r1cs: r1cs,
                                                         uniformityAnalysis: uniformity)
    let trace = engine.generateTraceCPU(r1cs: r1cs, witness: witness, layout: layout,
                                         transitionConstraints: constraints,
                                         uniformityAnalysis: uniformity)

    expectEqual(trace.count, layout.numColumns, "trace has correct number of columns")

    // Check A.w at row 0 = w[1] = 3
    let aVal = trace[layout.aValColumn][0]
    expect(frEqual(aVal, frFromInt(3)), "A.w[0] = 3")

    // Check B.w at row 0 = w[2] = 7
    let bVal = trace[layout.bValColumn][0]
    expect(frEqual(bVal, frFromInt(7)), "B.w[0] = 7")

    // Check C.w at row 0 = w[3] = 21
    let cVal = trace[layout.cValColumn][0]
    expect(frEqual(cVal, frFromInt(21)), "C.w[0] = 21")

    // Verify A*B = C at row 0
    let product = frMul(aVal, bVal)
    expect(frEqual(product, cVal), "A*B = C at row 0")
}

private func testCPUTraceAdditionGate() {
    let (r1cs, witness) = makeAdditionR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let wireAnalysis = engine.analyzeWireAccess(r1cs: r1cs)
    let uniformity = engine.analyzeUniformity(r1cs: r1cs)
    let layout = engine.computeLayout(r1cs: r1cs, wireAnalysis: wireAnalysis,
                                      uniformityAnalysis: uniformity)
    let constraints = engine.buildTransitionConstraints(r1cs: r1cs,
                                                         uniformityAnalysis: uniformity)
    let trace = engine.generateTraceCPU(r1cs: r1cs, witness: witness, layout: layout,
                                         transitionConstraints: constraints,
                                         uniformityAnalysis: uniformity)

    // A.w at row 0 = w[1] + w[2] = 5 + 12 = 17
    let aVal = trace[layout.aValColumn][0]
    expect(frEqual(aVal, frFromInt(17)), "A.w[0] = 17 (5+12)")

    // B.w at row 0 = w[0] = 1
    let bVal = trace[layout.bValColumn][0]
    expect(frEqual(bVal, Fr.one), "B.w[0] = 1")

    // C.w at row 0 = w[3] = 17
    let cVal = trace[layout.cValColumn][0]
    expect(frEqual(cVal, frFromInt(17)), "C.w[0] = 17")

    // A*B = 17*1 = 17 = C
    let product = frMul(aVal, bVal)
    expect(frEqual(product, cVal), "A*B = C for addition gate")
}

private func testCPUTraceMultiConstraint() {
    let (r1cs, witness) = makeMultiConstraintR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let wireAnalysis = engine.analyzeWireAccess(r1cs: r1cs)
    let uniformity = engine.analyzeUniformity(r1cs: r1cs)
    let layout = engine.computeLayout(r1cs: r1cs, wireAnalysis: wireAnalysis,
                                      uniformityAnalysis: uniformity)
    let constraints = engine.buildTransitionConstraints(r1cs: r1cs,
                                                         uniformityAnalysis: uniformity)
    let trace = engine.generateTraceCPU(r1cs: r1cs, witness: witness, layout: layout,
                                         transitionConstraints: constraints,
                                         uniformityAnalysis: uniformity)

    // Row 0: A.w = w[1]=3, B.w = w[2]=5, C.w = w[3]=15
    let aVal0 = trace[layout.aValColumn][0]
    let bVal0 = trace[layout.bValColumn][0]
    let cVal0 = trace[layout.cValColumn][0]
    expect(frEqual(frMul(aVal0, bVal0), cVal0), "constraint 0 satisfied")

    // Row 1: A.w = w[3]=15, B.w = w[4]=2, C.w = w[5]=30
    let aVal1 = trace[layout.aValColumn][1]
    let bVal1 = trace[layout.bValColumn][1]
    let cVal1 = trace[layout.cValColumn][1]
    expect(frEqual(frMul(aVal1, bVal1), cVal1), "constraint 1 satisfied")
}

private func testCPUTracePaddingZeros() {
    let (r1cs, witness) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let wireAnalysis = engine.analyzeWireAccess(r1cs: r1cs)
    let uniformity = engine.analyzeUniformity(r1cs: r1cs)
    let layout = engine.computeLayout(r1cs: r1cs, wireAnalysis: wireAnalysis,
                                      uniformityAnalysis: uniformity)
    let constraints = engine.buildTransitionConstraints(r1cs: r1cs,
                                                         uniformityAnalysis: uniformity)
    let trace = engine.generateTraceCPU(r1cs: r1cs, witness: witness, layout: layout,
                                         transitionConstraints: constraints,
                                         uniformityAnalysis: uniformity)

    // 1 constraint padded to 2 rows; row 1 of A/B/C should be zero
    if layout.numRows > 1 {
        let aValPad = trace[layout.aValColumn][1]
        expect(aValPad.isZero, "padded row has zero A value")
        let bValPad = trace[layout.bValColumn][1]
        expect(bValPad.isZero, "padded row has zero B value")
        let cValPad = trace[layout.cValColumn][1]
        expect(cValPad.isZero, "padded row has zero C value")
    }
}

// MARK: - Full Compilation Pipeline Tests

private func testCompileSimpleMultiplication() {
    let (r1cs, witness) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    do {
        let air = try engine.compile(r1cs: r1cs, witness: witness, useGPU: false)

        expectEqual(air.numR1CSConstraints, 1, "1 R1CS constraint")
        expectEqual(air.numR1CSVariables, 4, "4 variables")
        expectEqual(air.numPublicInputs, 1, "1 public input")
        expect(air.isUniform, "uniform AIR")
        expectEqual(air.maxDegree, 2, "max degree 2")

        // Verify trace
        let result = air.verify()
        expect(result == nil, "trace should verify: \(result ?? "ok")")
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

private func testCompileAdditionCircuit() {
    let (r1cs, witness) = makeAdditionR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    do {
        let air = try engine.compile(r1cs: r1cs, witness: witness, useGPU: false)
        let result = air.verify()
        expect(result == nil, "addition circuit verifies: \(result ?? "ok")")
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

private func testCompileMultiConstraintCircuit() {
    let (r1cs, witness) = makeMultiConstraintR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    do {
        let air = try engine.compile(r1cs: r1cs, witness: witness, useGPU: false)
        expectEqual(air.numR1CSConstraints, 2, "2 constraints")
        let result = air.verify()
        expect(result == nil, "multi-constraint circuit verifies: \(result ?? "ok")")
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

private func testCompileNonUniformCircuit() {
    let (r1cs, witness) = makeNonUniformR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    do {
        let air = try engine.compile(r1cs: r1cs, witness: witness, useGPU: false)
        expect(!air.isUniform, "non-uniform AIR")
        expectEqual(air.layout.numSelectorColumns, 2, "2 selector columns")
        let result = air.verify()
        expect(result == nil, "non-uniform circuit verifies: \(result ?? "ok")")
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

// MARK: - Trace Verification Tests

private func testVerifyValidTrace() {
    let (r1cs, witness) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    do {
        let air = try engine.compile(r1cs: r1cs, witness: witness, useGPU: false)
        let result = air.verify()
        expect(result == nil, "valid trace verifies")
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

private func testVerifyInvalidTrace() {
    let (r1cs, _) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    // Bad witness: 3 * 7 = 20 (should be 21)
    let a = frFromInt(3)
    let b = frFromInt(7)
    let badC = frFromInt(20)
    let badWitness = [Fr.one, a, b, badC]

    do {
        let air = try engine.compile(r1cs: r1cs, witness: badWitness, useGPU: false)
        let result = air.verify()
        expect(result != nil, "invalid trace should fail verification")
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

private func testVerifyBoundaryViolation() {
    let (r1cs, witness) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    do {
        var air = try engine.compile(r1cs: r1cs, witness: witness, useGPU: false)

        // Tamper with trace: change the constant-one wire column to have 0 instead of 1
        var tamperedTrace = air.trace
        let oneWireCol = air.layout.wireColumnOffset + 0
        if oneWireCol < tamperedTrace.count {
            tamperedTrace[oneWireCol][0] = Fr.zero
        }

        // Re-create with tampered trace
        let tampered = CompiledR1CSAIR(
            layout: air.layout,
            transitionConstraints: air.transitionConstraints,
            boundaryConstraints: air.boundaryConstraints,
            trace: tamperedTrace,
            isUniform: air.isUniform,
            maxDegree: air.maxDegree,
            numR1CSConstraints: air.numR1CSConstraints,
            numR1CSVariables: air.numR1CSVariables,
            numPublicInputs: air.numPublicInputs,
            wireAccessCounts: air.wireAccessCounts
        )

        let result = tampered.verify()
        expect(result != nil, "boundary violation detected: \(result ?? "none")")
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

// MARK: - Batch Constraint Evaluation Tests

private func testBatchEvalSatisfied() {
    let (r1cs, witness) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    do {
        let air = try engine.compile(r1cs: r1cs, witness: witness, useGPU: false)
        let evalResult = engine.evaluateConstraints(
            trace: air.trace, layout: air.layout,
            transitionConstraints: air.transitionConstraints,
            numActiveRows: air.numR1CSConstraints)

        expect(evalResult.allSatisfied, "all constraints satisfied")
        expectEqual(evalResult.violations.count, 0, "no violations")
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

private func testBatchEvalUnsatisfied() {
    let (r1cs, _) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    let a = frFromInt(3)
    let b = frFromInt(7)
    let badC = frFromInt(20)
    let badWitness = [Fr.one, a, b, badC]

    do {
        let air = try engine.compile(r1cs: r1cs, witness: badWitness, useGPU: false)
        let evalResult = engine.evaluateConstraints(
            trace: air.trace, layout: air.layout,
            transitionConstraints: air.transitionConstraints,
            numActiveRows: air.numR1CSConstraints)

        expect(!evalResult.allSatisfied, "unsatisfied constraints detected")
        expect(evalResult.violations.count > 0, "at least 1 violation")
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

private func testBatchEvalWithResiduals() {
    let (r1cs, witness) = makeMultiConstraintR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    do {
        let air = try engine.compile(r1cs: r1cs, witness: witness, useGPU: false)
        let evalResult = engine.evaluateConstraints(
            trace: air.trace, layout: air.layout,
            transitionConstraints: air.transitionConstraints,
            numActiveRows: air.numR1CSConstraints,
            collectResiduals: true)

        expect(evalResult.allSatisfied, "all constraints satisfied")
        // Residuals should be collected
        expect(evalResult.residuals.count > 0, "residuals collected")
        // All residuals should be zero for satisfied circuit
        for ci in 0..<evalResult.residuals.count {
            for row in 0..<air.numR1CSConstraints {
                expect(evalResult.residuals[ci][row].isZero,
                       "residual[\(ci)][\(row)] is zero")
            }
        }
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

// MARK: - FrAIRExpression Conversion Tests

private func testExpressionSingleTerm() {
    let (r1cs, witness) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    do {
        let air = try engine.compile(r1cs: r1cs, witness: witness, useGPU: false)
        let expressions = engine.toFrAIRExpressions(air: air)

        expectEqual(expressions.count, 1, "1 expression for 1 constraint")
        // Degree should be 2 (A * B - C)
        expectEqual(expressions[0].degree, 2, "expression degree is 2")
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

private func testExpressionMultiTerm() {
    let (r1cs, witness) = makeAdditionR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    do {
        let air = try engine.compile(r1cs: r1cs, witness: witness, useGPU: false)
        let expressions = engine.toFrAIRExpressions(air: air)

        expectEqual(expressions.count, 1, "1 expression")
        // A has 2 terms, so the A expression has degree 1 (linear)
        // A*B has degree 2, A*B - C has degree 2
        expectEqual(expressions[0].degree, 2, "expression degree is 2")
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

private func testExpressionMultiConstraint() {
    let (r1cs, witness) = makeNonUniformR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    do {
        let air = try engine.compile(r1cs: r1cs, witness: witness, useGPU: false)
        let expressions = engine.toFrAIRExpressions(air: air)

        expectEqual(expressions.count, 2, "2 expressions for non-uniform")
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

// MARK: - Density Stats Tests

private func testDensitySparse() {
    let (r1cs, _) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let stats = engine.computeDensityStats(r1cs: r1cs)

    expectEqual(stats.totalNonZeros, 3, "3 non-zero entries total")
    expect(stats.aDensity > 0 && stats.aDensity < 1.0, "A is sparse")
    expect(stats.avgEntriesPerRow > 2.9, "~3 entries per row")
}

private func testDensityDense() {
    // Create a fully connected constraint: all wires appear in A, B, C
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()
    for col in 0..<3 {
        aE.append(R1CSEntry(row: 0, col: col, val: one))
        bE.append(R1CSEntry(row: 0, col: col, val: one))
        cE.append(R1CSEntry(row: 0, col: col, val: one))
    }
    let r1cs = R1CSInstance(numConstraints: 1, numVars: 3, numPublic: 0,
                            aEntries: aE, bEntries: bE, cEntries: cE)
    let engine = GPUR1CSToAIRCompilerEngine()
    let stats = engine.computeDensityStats(r1cs: r1cs)

    expectEqual(stats.totalNonZeros, 9, "9 non-zero entries")
    // 3 entries / (1 * 3) = 1.0 density per matrix
    expect(stats.aDensity > 0.99, "A is fully dense")
}

// MARK: - Vanishing Polynomial Tests

private func testVanishingAtRootOfUnity() {
    let engine = GPUR1CSToAIRCompilerEngine()
    // omega = primitive 2^1-th root of unity (i.e., -1 in the field)
    // Z_H(omega) = omega^2 - 1 = 1 - 1 = 0
    let logN = 1
    let omega = computeNthRootOfUnity(logN: logN)
    let vanishing = engine.evaluateVanishingPoly(at: omega, logTraceLength: logN)
    expect(vanishing.isZero, "vanishing poly is zero at root of unity")
}

private func testVanishingAtRandomPoint() {
    let engine = GPUR1CSToAIRCompilerEngine()
    // Evaluate at a non-root point (x = 2)
    let x = frFromInt(2)
    let logN = 2 // domain size 4
    let vanishing = engine.evaluateVanishingPoly(at: x, logTraceLength: logN)
    // Z_H(2) = 2^4 - 1 = 15
    let expected = frFromInt(15)
    expect(frEqual(vanishing, expected), "Z_H(2) = 15 for n=4")
}

// MARK: - Constraint Composition Tests

private func testR1CSAIRCompositionSingleConstraint() {
    let (r1cs, witness) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    do {
        let air = try engine.compile(r1cs: r1cs, witness: witness, useGPU: false)
        let alpha = frFromInt(7)
        let composed = engine.composeConstraints(
            trace: air.trace, layout: air.layout,
            transitionConstraints: air.transitionConstraints,
            alpha: alpha, numActiveRows: air.numR1CSConstraints)

        // For satisfied circuit, all composed values should be zero
        for (i, val) in composed.enumerated() {
            expect(val.isZero, "composed[\(i)] is zero for satisfied circuit")
        }
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

private func testR1CSAIRCompositionMultipleConstraints() {
    let (r1cs, witness) = makeNonUniformR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    do {
        let air = try engine.compile(r1cs: r1cs, witness: witness, useGPU: false)
        let alpha = frFromInt(13)
        let composed = engine.composeConstraints(
            trace: air.trace, layout: air.layout,
            transitionConstraints: air.transitionConstraints,
            alpha: alpha, numActiveRows: air.numR1CSConstraints)

        for (i, val) in composed.enumerated() {
            expect(val.isZero, "composed[\(i)] is zero for satisfied non-uniform")
        }
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

// MARK: - Selector Polynomial Tests

private func testSelectorUniform() {
    let engine = GPUR1CSToAIRCompilerEngine()
    let analysis = R1CSUniformityAnalysis(
        isUniform: true, numDistinctStructures: 1,
        constraintStructureMap: [0, 0], structures: [(1, 1, 1)])
    let selectors = engine.buildSelectorPolynomials(
        uniformityAnalysis: analysis, logTraceLength: 2)

    expectEqual(selectors.count, 1, "1 selector for uniform")
    // Both rows active for the single structure
    expect(frEqual(selectors[0][0], Fr.one), "selector[0][0] = 1")
    expect(frEqual(selectors[0][1], Fr.one), "selector[0][1] = 1")
}

private func testSelectorNonUniform() {
    let engine = GPUR1CSToAIRCompilerEngine()
    let analysis = R1CSUniformityAnalysis(
        isUniform: false, numDistinctStructures: 2,
        constraintStructureMap: [0, 1], structures: [(1, 1, 1), (2, 1, 1)])
    let selectors = engine.buildSelectorPolynomials(
        uniformityAnalysis: analysis, logTraceLength: 2)

    expectEqual(selectors.count, 2, "2 selectors for 2 structures")
    // Structure 0 active at row 0 only
    expect(frEqual(selectors[0][0], Fr.one), "sel[0][0] = 1")
    expect(selectors[0][1].isZero, "sel[0][1] = 0")
    // Structure 1 active at row 1 only
    expect(selectors[1][0].isZero, "sel[1][0] = 0")
    expect(frEqual(selectors[1][1], Fr.one), "sel[1][1] = 1")
}

// MARK: - Quick Verify Tests

private func testQuickVerifyValid() {
    let (r1cs, witness) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    do {
        let air = try engine.compile(r1cs: r1cs, witness: witness, useGPU: false)
        let ok = engine.quickVerify(trace: air.trace, layout: air.layout,
                                    numConstraints: air.numR1CSConstraints)
        expect(ok, "quick verify passes for valid trace")
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

private func testQuickVerifyInvalid() {
    let (r1cs, _) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    let badWitness = [Fr.one, frFromInt(3), frFromInt(7), frFromInt(20)]

    do {
        let air = try engine.compile(r1cs: r1cs, witness: badWitness, useGPU: false)
        let ok = engine.quickVerify(trace: air.trace, layout: air.layout,
                                    numConstraints: air.numR1CSConstraints)
        expect(!ok, "quick verify fails for invalid trace")
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

// MARK: - Public Input Extraction Tests

private func testExtractPublicInputs() {
    let (r1cs, witness) = makeMultiConstraintR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    do {
        let air = try engine.compile(r1cs: r1cs, witness: witness, useGPU: false)
        let pubInputs = engine.extractPublicInputs(air: air)

        expectEqual(pubInputs.count, 2, "2 public inputs extracted")
        expect(frEqual(pubInputs[0], witness[1]), "first public input = witness[1]")
        expect(frEqual(pubInputs[1], witness[2]), "second public input = witness[2]")
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

// MARK: - Column Reorder Tests

private func testColumnReorderIdentity() {
    let engine = GPUR1CSToAIRCompilerEngine()
    let trace: [[Fr]] = [
        [frFromInt(1), frFromInt(2)],
        [frFromInt(3), frFromInt(4)],
        [frFromInt(5), frFromInt(6)]
    ]
    let permutation = [0, 1, 2]
    let reordered = engine.reorderTraceColumns(trace, permutation: permutation)

    expectEqual(reordered.count, 3, "same number of columns")
    expect(frEqual(reordered[0][0], frFromInt(1)), "identity reorder preserves col 0")
    expect(frEqual(reordered[2][1], frFromInt(6)), "identity reorder preserves col 2")
}

private func testColumnReorderSwap() {
    let engine = GPUR1CSToAIRCompilerEngine()
    let trace: [[Fr]] = [
        [frFromInt(1), frFromInt(2)],
        [frFromInt(3), frFromInt(4)],
        [frFromInt(5), frFromInt(6)]
    ]
    let permutation = [2, 0, 1] // swap columns
    let reordered = engine.reorderTraceColumns(trace, permutation: permutation)

    // New col 0 = old col 2
    expect(frEqual(reordered[0][0], frFromInt(5)), "reorder[0] = old col 2")
    // New col 1 = old col 0
    expect(frEqual(reordered[1][0], frFromInt(1)), "reorder[1] = old col 0")
    // New col 2 = old col 1
    expect(frEqual(reordered[2][0], frFromInt(3)), "reorder[2] = old col 1")
}

// MARK: - Degree Analysis Tests

private func testDegreeAnalysis() {
    let engine = GPUR1CSToAIRCompilerEngine()
    let constraints = [
        R1CSAIRTransitionConstraint(constraintIndex: 0,
                                     aTerms: [(0, Fr.one)],
                                     bTerms: [(1, Fr.one)],
                                     cTerms: [(2, Fr.one)]),
        R1CSAIRTransitionConstraint(constraintIndex: 1,
                                     aTerms: [(0, Fr.one), (1, Fr.one)],
                                     bTerms: [(2, Fr.one)],
                                     cTerms: [(3, Fr.one)])
    ]

    let analysis = engine.analyzeDegrees(
        transitionConstraints: constraints, logTraceLength: 3)

    expectEqual(analysis.maxTransitionDegree, 2, "max transition degree is 2")
    expectEqual(analysis.transitionDegrees.count, 2, "2 constraint degrees")
    // Composition degree bound = maxDegree * traceLength = 2 * 8 = 16
    expectEqual(analysis.compositionDegreeBound, 16,
                "composition degree bound = 2 * 8")
    expectEqual(analysis.numQuotientChunks, 2, "2 quotient chunks")
}

// MARK: - Error Handling Tests

private func testInsufficientWitness() {
    let (r1cs, _) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()
    let shortWitness = [Fr.one, frFromInt(3)] // Only 2 elements, need 4

    var didThrow = false
    do {
        _ = try engine.compile(r1cs: r1cs, witness: shortWitness)
    } catch {
        didThrow = true
    }
    expect(didThrow, "insufficient witness throws error")
}

// MARK: - GPU Trace Generation Tests

private func testGPUTraceMultiplicationGate() {
    let (r1cs, witness) = makeMultiplicationR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    do {
        // Use GPU path (will fall back to CPU if no GPU available)
        let air = try engine.compile(r1cs: r1cs, witness: witness, useGPU: true)

        // Verify same results as CPU path
        let aVal = air.trace[air.layout.aValColumn][0]
        let bVal = air.trace[air.layout.bValColumn][0]
        let cVal = air.trace[air.layout.cValColumn][0]

        expect(frEqual(aVal, frFromInt(3)), "GPU: A.w[0] = 3")
        expect(frEqual(bVal, frFromInt(7)), "GPU: B.w[0] = 7")
        expect(frEqual(cVal, frFromInt(21)), "GPU: C.w[0] = 21")

        let product = frMul(aVal, bVal)
        expect(frEqual(product, cVal), "GPU: A*B = C at row 0")

        let result = air.verify()
        expect(result == nil, "GPU trace verifies: \(result ?? "ok")")
    } catch {
        expect(false, "compile threw: \(error)")
    }
}

private func testGPUTraceMultiConstraint() {
    let (r1cs, witness) = makeMultiConstraintR1CS()
    let engine = GPUR1CSToAIRCompilerEngine()

    do {
        let air = try engine.compile(r1cs: r1cs, witness: witness, useGPU: true)

        // Verify both constraint rows
        for row in 0..<2 {
            let aVal = air.trace[air.layout.aValColumn][row]
            let bVal = air.trace[air.layout.bValColumn][row]
            let cVal = air.trace[air.layout.cValColumn][row]
            let product = frMul(aVal, bVal)
            expect(frEqual(product, cVal), "GPU: constraint \(row) satisfied")
        }

        let result = air.verify()
        expect(result == nil, "GPU multi-constraint trace verifies: \(result ?? "ok")")
    } catch {
        expect(false, "compile threw: \(error)")
    }
}
