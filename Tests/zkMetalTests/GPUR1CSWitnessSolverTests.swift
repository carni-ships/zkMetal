// GPU R1CS Witness Solver Engine tests
import zkMetal
import Foundation

private func frEq(_ a: Fr, _ b: Fr) -> Bool {
    return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
           a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

public func runGPUR1CSWitnessSolverTests() {
    testSimpleLinearWitnessSolve()
    testMultiplicationWitnessSolve()
    testChainedConstraintSolve()
    testTopologicalOrdering()
    testHintBasedSolving()
    testDivisionHint()
    testWitnessValidation()
    testWitnessValidationFailure()
    testIncrementalSolving()
    testBatchFieldMul()
    testBatchFieldAdd()
    testBatchFieldSub()
    testSolveFromR1CSInstance()
    testDependencyAnalysis()
    testMakeConstraintHelper()
    testUnresolvableSystem()
    testMultipleHints()
    testLargerCircuit()
    testEmptySystem()
    testSinglePublicInput()
    testComparisonHint()
    testIncrementalMultiStage()
}

// MARK: - Test 1: Simple linear witness solve

private func testSimpleLinearWitnessSolve() {
    suite("Witness Solver: simple linear")

    // Circuit: (a + b) * 1 = c
    // Wires: [1, a, b, c]  (a, b public => numPublic = 2)
    let engine = GPUR1CSWitnessSolverEngine()

    let c0 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 1, val: Fr.one), (col: 2, val: Fr.one)],
        b: [(col: 0, val: Fr.one)],
        c: [(col: 3, val: Fr.one)],
        index: 0
    )

    let system = GPUR1CSWitnessSolverEngine.makeSystem(
        constraints: [c0], numVars: 4, numPublic: 2
    )

    let a = frFromInt(10)
    let b = frFromInt(20)
    let result = engine.solve(system: system, publicInputs: [a, b])

    expect(result.isComplete, "all variables resolved")
    expectEqual(result.unresolvedIndices.count, 0, "no unresolved")
    expect(frEq(result.witness[0], Fr.one), "wire 0 = 1")
    expect(frEq(result.witness[1], a), "wire 1 = a")
    expect(frEq(result.witness[2], b), "wire 2 = b")
    expect(frEq(result.witness[3], frAdd(a, b)), "wire 3 = a + b = 30")
}

// MARK: - Test 2: Multiplication witness solve

private func testMultiplicationWitnessSolve() {
    suite("Witness Solver: multiplication")

    // Circuit: a * b = c
    // Wires: [1, a, b, c]  (a, b public => numPublic = 2)
    let engine = GPUR1CSWitnessSolverEngine()

    let c0 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 1, val: Fr.one)],
        b: [(col: 2, val: Fr.one)],
        c: [(col: 3, val: Fr.one)],
        index: 0
    )

    let system = GPUR1CSWitnessSolverEngine.makeSystem(
        constraints: [c0], numVars: 4, numPublic: 2
    )

    let a = frFromInt(7)
    let b = frFromInt(11)
    let result = engine.solve(system: system, publicInputs: [a, b])

    expect(result.isComplete, "multiplication solved")
    let expected = frMul(a, b) // 77
    expect(frEq(result.witness[3], expected), "wire 3 = a * b = 77")

    // Validate the witness
    let validation = engine.validate(system: system, witness: result.witness)
    expect(validation.valid, "witness validates")
    expectEqual(validation.violatedConstraints.count, 0, "no violated constraints")
}

// MARK: - Test 3: Chained constraint solve

private func testChainedConstraintSolve() {
    suite("Witness Solver: chained constraints")

    // Circuit: a * b = c, c * d = e
    // Wires: [1, a, b, d, c, e]  (a, b, d public => numPublic = 3)
    let engine = GPUR1CSWitnessSolverEngine()

    // Constraint 0: w[1] * w[2] = w[4]
    let c0 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 1, val: Fr.one)],
        b: [(col: 2, val: Fr.one)],
        c: [(col: 4, val: Fr.one)],
        index: 0
    )

    // Constraint 1: w[4] * w[3] = w[5]
    let c1 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 4, val: Fr.one)],
        b: [(col: 3, val: Fr.one)],
        c: [(col: 5, val: Fr.one)],
        index: 1
    )

    let system = GPUR1CSWitnessSolverEngine.makeSystem(
        constraints: [c0, c1], numVars: 6, numPublic: 3
    )

    let a = frFromInt(3)
    let b = frFromInt(5)
    let d = frFromInt(2)
    let result = engine.solve(system: system, publicInputs: [a, b, d])

    expect(result.isComplete, "chained solve complete")
    let c = frMul(a, b)    // 15
    let e = frMul(c, d)    // 30
    expect(frEq(result.witness[4], c), "wire 4 = a*b = 15")
    expect(frEq(result.witness[5], e), "wire 5 = c*d = 30")
    expect(result.iterations >= 1, "at least 1 iteration")
}

// MARK: - Test 4: Topological ordering

private func testTopologicalOrdering() {
    suite("Witness Solver: topological order")

    let engine = GPUR1CSWitnessSolverEngine()

    // Constraints in "wrong" order: c1 depends on c0's output
    // c1: w[4] * w[3] = w[5] (needs w[4] from c0)
    // c0: w[1] * w[2] = w[4] (can solve immediately from public inputs)
    let c1 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 4, val: Fr.one)],
        b: [(col: 3, val: Fr.one)],
        c: [(col: 5, val: Fr.one)],
        index: 1
    )
    let c0 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 1, val: Fr.one)],
        b: [(col: 2, val: Fr.one)],
        c: [(col: 4, val: Fr.one)],
        index: 0
    )

    let system = GPUR1CSWitnessSolverEngine.makeSystem(
        constraints: [c1, c0], numVars: 6, numPublic: 3
    )

    // Known vars: 0 (one), 1, 2, 3 (public inputs)
    let knownVars: Set<Int> = [0, 1, 2, 3]
    let order = engine.topologicalOrder(system: system, knownVars: knownVars)

    expectEqual(order.count, 2, "2 constraints in order")
    // c0 (index 0) should come before c1 (index 1) since c0 is solvable first
    if order.count == 2 {
        expectEqual(order[0], 0, "c0 first (produces w[4])")
        expectEqual(order[1], 1, "c1 second (needs w[4])")
    }
}

// MARK: - Test 5: Hint-based solving

private func testHintBasedSolving() {
    suite("Witness Solver: hint-based")

    // Circuit with a non-linear constraint that needs a hint:
    // w[1] * w[2] = w[3]  where w[2] is private and needs a hint
    // Wires: [1, a, secret, product]  (a public => numPublic = 1)
    let engine = GPUR1CSWitnessSolverEngine()

    let c0 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 1, val: Fr.one)],
        b: [(col: 2, val: Fr.one)],
        c: [(col: 3, val: Fr.one)],
        index: 0
    )

    let system = GPUR1CSWitnessSolverEngine.makeSystem(
        constraints: [c0], numVars: 4, numPublic: 1
    )

    let a = frFromInt(6)
    let secret = frFromInt(7)

    // Provide hint for the secret variable
    let hint = WitnessHint(targetVar: 2) { _ in secret }
    let result = engine.solve(system: system, publicInputs: [a], hints: [hint])

    expect(result.isComplete, "hint-based solve complete")
    expect(frEq(result.witness[2], secret), "wire 2 = secret = 7")
    expect(frEq(result.witness[3], frMul(a, secret)), "wire 3 = a * secret = 42")
    expectEqual(result.hintSolvedCount, 1, "1 variable solved by hint")
}

// MARK: - Test 6: Division hint

private func testDivisionHint() {
    suite("Witness Solver: division hint")

    // Circuit: a * b_inv = 1 (proving knowledge of inverse)
    // Actually encoded as: a * b_inv = w[0] (the constant 1 wire)
    // Wires: [1, a, b_inv]  (a public => numPublic = 1)
    let engine = GPUR1CSWitnessSolverEngine()

    let c0 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 1, val: Fr.one)],
        b: [(col: 2, val: Fr.one)],
        c: [(col: 0, val: Fr.one)],  // = 1
        index: 0
    )

    let system = GPUR1CSWitnessSolverEngine.makeSystem(
        constraints: [c0], numVars: 3, numPublic: 1
    )

    let a = frFromInt(5)

    // Hint: compute 1/a
    let divHint = WitnessHint(targetVar: 2) { w in
        frInverse(w[1])
    }

    let result = engine.solve(system: system, publicInputs: [a], hints: [divHint])
    expect(result.isComplete, "division solve complete")

    // Verify: a * (1/a) = 1
    let product = frMul(result.witness[1], result.witness[2])
    expect(frEq(product, Fr.one), "a * a_inv = 1")
}

// MARK: - Test 7: Witness validation (valid)

private func testWitnessValidation() {
    suite("Witness Solver: validation pass")

    let engine = GPUR1CSWitnessSolverEngine()

    // System: a * b = c
    let c0 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 1, val: Fr.one)],
        b: [(col: 2, val: Fr.one)],
        c: [(col: 3, val: Fr.one)],
        index: 0
    )

    let system = GPUR1CSWitnessSolverEngine.makeSystem(
        constraints: [c0], numVars: 4, numPublic: 2
    )

    let a = frFromInt(4)
    let b = frFromInt(9)
    let c = frMul(a, b)
    let witness = [Fr.one, a, b, c]

    let validation = engine.validate(system: system, witness: witness)
    expect(validation.valid, "valid witness passes validation")
    expectEqual(validation.numConstraints, 1, "1 constraint")
    expectEqual(validation.violatedConstraints.count, 0, "no violations")
    expect(frEq(validation.residuals[0], Fr.zero), "residual is zero")
}

// MARK: - Test 8: Witness validation (failure)

private func testWitnessValidationFailure() {
    suite("Witness Solver: validation fail")

    let engine = GPUR1CSWitnessSolverEngine()

    let c0 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 1, val: Fr.one)],
        b: [(col: 2, val: Fr.one)],
        c: [(col: 3, val: Fr.one)],
        index: 0
    )

    let system = GPUR1CSWitnessSolverEngine.makeSystem(
        constraints: [c0], numVars: 4, numPublic: 2
    )

    let a = frFromInt(4)
    let b = frFromInt(9)
    let wrongC = frFromInt(99)  // should be 36
    let witness = [Fr.one, a, b, wrongC]

    let validation = engine.validate(system: system, witness: witness)
    expect(!validation.valid, "invalid witness fails validation")
    expectEqual(validation.violatedConstraints.count, 1, "1 violated constraint")
    expectEqual(validation.violatedConstraints[0], 0, "constraint 0 violated")
    expect(!frEq(validation.residuals[0], Fr.zero), "residual is nonzero")

    // Verify residual = a*b - wrongC = 36 - 99
    let expectedResidual = frSub(frMul(a, b), wrongC)
    expect(frEq(validation.residuals[0], expectedResidual), "residual = a*b - wrongC")
}

// MARK: - Test 9: Incremental solving

private func testIncrementalSolving() {
    suite("Witness Solver: incremental")

    let engine = GPUR1CSWitnessSolverEngine()

    // Start with: a * b = c
    // Wires: [1, a, b, c, d, e]  (a, b public => numPublic = 2)
    let a = frFromInt(3)
    let b = frFromInt(4)

    var state = engine.makeIncrementalState(numVars: 6, numPublic: 2,
                                             publicInputs: [a, b])

    // Phase 1: add a * b = c
    let c0 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 1, val: Fr.one)],
        b: [(col: 2, val: Fr.one)],
        c: [(col: 3, val: Fr.one)],
        index: 0
    )

    let r1 = engine.solveIncremental(state: &state, newConstraints: [c0])
    let c = frMul(a, b)
    expect(frEq(state.witness[3], c), "phase 1: wire 3 = a*b = 12")

    // Phase 2: add (c + 1) * 1 = d  => d = c + 1 = 13
    // Encoded as: (w[3] + w[0]) * w[0] = w[4]
    let c1 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 3, val: Fr.one), (col: 0, val: Fr.one)],
        b: [(col: 0, val: Fr.one)],
        c: [(col: 4, val: Fr.one)],
        index: 1
    )

    let r2 = engine.solveIncremental(state: &state, newConstraints: [c1])
    let d = frAdd(c, Fr.one)
    expect(frEq(state.witness[4], d), "phase 2: wire 4 = c+1 = 13")

    _ = r1
    _ = r2
}

// MARK: - Test 10: Batch field multiply

private func testBatchFieldMul() {
    suite("Witness Solver: batch field mul")

    let engine = GPUR1CSWitnessSolverEngine()
    let n = 64

    var a = [Fr]()
    var b = [Fr]()
    var expected = [Fr]()

    for i in 0..<n {
        let ai = frFromInt(UInt64(i + 1))
        let bi = frFromInt(UInt64(i + 100))
        a.append(ai)
        b.append(bi)
        expected.append(frMul(ai, bi))
    }

    let result = engine.batchFieldMul(a, b)
    expectEqual(result.count, n, "batch mul count")

    var allMatch = true
    for i in 0..<n {
        if !frEq(result[i], expected[i]) {
            allMatch = false
            break
        }
    }
    expect(allMatch, "all batch mul results match")
}

// MARK: - Test 11: Batch field add

private func testBatchFieldAdd() {
    suite("Witness Solver: batch field add")

    let engine = GPUR1CSWitnessSolverEngine()
    let n = 64

    var a = [Fr]()
    var b = [Fr]()
    var expected = [Fr]()

    for i in 0..<n {
        let ai = frFromInt(UInt64(i * 3 + 1))
        let bi = frFromInt(UInt64(i * 7 + 2))
        a.append(ai)
        b.append(bi)
        expected.append(frAdd(ai, bi))
    }

    let result = engine.batchFieldAdd(a, b)
    expectEqual(result.count, n, "batch add count")

    var allMatch = true
    for i in 0..<n {
        if !frEq(result[i], expected[i]) {
            allMatch = false
            break
        }
    }
    expect(allMatch, "all batch add results match")
}

// MARK: - Test 12: Batch field subtract

private func testBatchFieldSub() {
    suite("Witness Solver: batch field sub")

    let engine = GPUR1CSWitnessSolverEngine()
    let n = 64

    var a = [Fr]()
    var b = [Fr]()
    var expected = [Fr]()

    for i in 0..<n {
        let ai = frFromInt(UInt64(i * 10 + 100))
        let bi = frFromInt(UInt64(i * 3 + 1))
        a.append(ai)
        b.append(bi)
        expected.append(frSub(ai, bi))
    }

    let result = engine.batchFieldSub(a, b)
    expectEqual(result.count, n, "batch sub count")

    var allMatch = true
    for i in 0..<n {
        if !frEq(result[i], expected[i]) {
            allMatch = false
            break
        }
    }
    expect(allMatch, "all batch sub results match")
}

// MARK: - Test 13: Solve from R1CSInstance

private func testSolveFromR1CSInstance() {
    suite("Witness Solver: solve from R1CSInstance")

    let engine = GPUR1CSWitnessSolverEngine()

    // Build an R1CSInstance: a * b = c
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()
    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    bE.append(R1CSEntry(row: 0, col: 2, val: one))
    cE.append(R1CSEntry(row: 0, col: 3, val: one))

    let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 2,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    let a = frFromInt(8)
    let b = frFromInt(13)
    let result = engine.solveFromR1CS(r1cs: r1cs, publicInputs: [a, b])

    expect(result.isComplete, "R1CSInstance solve complete")
    expect(frEq(result.witness[3], frMul(a, b)), "wire 3 = 8 * 13 = 104")

    // Also validate via R1CS validation
    let validation = engine.validateR1CS(r1cs: r1cs, witness: result.witness)
    expect(validation.valid, "validates against R1CSInstance")
}

// MARK: - Test 14: Dependency analysis

private func testDependencyAnalysis() {
    suite("Witness Solver: dependency analysis")

    let engine = GPUR1CSWitnessSolverEngine()

    // c0: w[1] * w[2] = w[3]
    // c1: w[3] * w[1] = w[4]
    let c0 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 1, val: Fr.one)],
        b: [(col: 2, val: Fr.one)],
        c: [(col: 3, val: Fr.one)],
        index: 0
    )
    let c1 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 3, val: Fr.one)],
        b: [(col: 1, val: Fr.one)],
        c: [(col: 4, val: Fr.one)],
        index: 1
    )

    let system = GPUR1CSWitnessSolverEngine.makeSystem(
        constraints: [c0, c1], numVars: 5, numPublic: 2
    )

    let deps = engine.analyzeDependencies(system: system)
    expectEqual(deps.count, 2, "2 constraint dependencies")

    // c0: inputs = {1, 2}, outputs = {3}
    expect(deps[0].inputs.contains(1), "c0 input: 1")
    expect(deps[0].inputs.contains(2), "c0 input: 2")
    expect(deps[0].outputs.contains(3), "c0 output: 3")

    // c1: inputs = {3, 1}, outputs = {4}
    expect(deps[1].inputs.contains(3), "c1 input: 3")
    expect(deps[1].inputs.contains(1), "c1 input: 1")
    expect(deps[1].outputs.contains(4), "c1 output: 4")
}

// MARK: - Test 15: Make constraint helper

private func testMakeConstraintHelper() {
    suite("Witness Solver: makeConstraint helper")

    let two = frFromInt(2)
    let three = frFromInt(3)

    let c = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 1, val: two), (col: 2, val: three)],
        b: [(col: 0, val: Fr.one)],
        c: [(col: 3, val: Fr.one)],
        index: 7
    )

    expectEqual(c.index, 7, "constraint index = 7")
    expectEqual(c.a.terms.count, 2, "A has 2 terms")
    expectEqual(c.b.terms.count, 1, "B has 1 term")
    expectEqual(c.c.terms.count, 1, "C has 1 term")
    expectEqual(c.a.terms[0].varIdx, 1, "A term 0 var = 1")
    expect(frEq(c.a.terms[0].coeff, two), "A term 0 coeff = 2")
    expectEqual(c.a.terms[1].varIdx, 2, "A term 1 var = 2")
    expect(frEq(c.a.terms[1].coeff, three), "A term 1 coeff = 3")
}

// MARK: - Test 16: Unresolvable system

private func testUnresolvableSystem() {
    suite("Witness Solver: unresolvable system")

    let engine = GPUR1CSWitnessSolverEngine()

    // w[1] * w[2] = w[3], but w[1] and w[2] are both private (unknown)
    // With only numPublic = 0, nothing can be solved
    let c0 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 1, val: Fr.one)],
        b: [(col: 2, val: Fr.one)],
        c: [(col: 3, val: Fr.one)],
        index: 0
    )

    let system = GPUR1CSWitnessSolverEngine.makeSystem(
        constraints: [c0], numVars: 4, numPublic: 0
    )

    let result = engine.solve(system: system, publicInputs: [])
    expect(!result.isComplete, "system is not fully solvable")
    expect(result.unresolvedIndices.count > 0, "has unresolved variables")
}

// MARK: - Test 17: Multiple hints

private func testMultipleHints() {
    suite("Witness Solver: multiple hints")

    let engine = GPUR1CSWitnessSolverEngine()

    // w[1] * w[2] = w[3]; all private, solve with 2 hints
    let c0 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 1, val: Fr.one)],
        b: [(col: 2, val: Fr.one)],
        c: [(col: 3, val: Fr.one)],
        index: 0
    )

    let system = GPUR1CSWitnessSolverEngine.makeSystem(
        constraints: [c0], numVars: 4, numPublic: 0
    )

    let h1 = WitnessHint(targetVar: 1) { _ in frFromInt(5) }
    let h2 = WitnessHint(targetVar: 2) { _ in frFromInt(9) }

    let result = engine.solve(system: system, publicInputs: [], hints: [h1, h2])
    expect(result.isComplete, "fully solved with 2 hints")
    expect(frEq(result.witness[1], frFromInt(5)), "wire 1 = 5")
    expect(frEq(result.witness[2], frFromInt(9)), "wire 2 = 9")
    expect(frEq(result.witness[3], frMul(frFromInt(5), frFromInt(9))), "wire 3 = 45")
    expectEqual(result.hintSolvedCount, 2, "2 hint-solved variables")
}

// MARK: - Test 18: Larger circuit (Fibonacci-like)

private func testLargerCircuit() {
    suite("Witness Solver: larger circuit")

    let engine = GPUR1CSWitnessSolverEngine()

    // Fibonacci-like: a[i+2] = a[i] + a[i+1] encoded as
    // (w[i] + w[i+1]) * 1 = w[i+2]
    // Wires: [1, fib0, fib1, fib2, fib3, fib4, fib5, fib6, fib7]
    // 9 wires, 7 constraints, 2 public inputs (fib0, fib1)
    let numWires = 9
    let numConstraints = 7

    var constraints = [WitnessConstraint]()
    for i in 0..<numConstraints {
        let c = GPUR1CSWitnessSolverEngine.makeConstraint(
            a: [(col: 1 + i, val: Fr.one), (col: 2 + i, val: Fr.one)],
            b: [(col: 0, val: Fr.one)],
            c: [(col: 3 + i, val: Fr.one)],
            index: i
        )
        constraints.append(c)
    }

    let system = GPUR1CSWitnessSolverEngine.makeSystem(
        constraints: constraints, numVars: numWires, numPublic: 2
    )

    // Fibonacci starting at 1, 1: 1, 1, 2, 3, 5, 8, 13, 21
    let result = engine.solve(system: system, publicInputs: [Fr.one, Fr.one])

    expect(result.isComplete, "fibonacci circuit fully solved")

    // Verify: 1, 1, 2, 3, 5, 8, 13, 21
    let expectedVals: [UInt64] = [1, 1, 2, 3, 5, 8, 13, 21]
    for i in 0..<expectedVals.count {
        let expected = frFromInt(expectedVals[i])
        expect(frEq(result.witness[1 + i], expected),
               "fib[\(i)] = \(expectedVals[i])")
    }

    // Validate
    let validation = engine.validate(system: system, witness: result.witness)
    expect(validation.valid, "fibonacci witness validates")
}

// MARK: - Test 19: Empty system

private func testEmptySystem() {
    suite("Witness Solver: empty system")

    let engine = GPUR1CSWitnessSolverEngine()

    let system = GPUR1CSWitnessSolverEngine.makeSystem(
        constraints: [], numVars: 3, numPublic: 2
    )

    let a = frFromInt(42)
    let b = frFromInt(99)
    let result = engine.solve(system: system, publicInputs: [a, b])

    // With no constraints, only wire 0 and public inputs are known
    expect(frEq(result.witness[0], Fr.one), "wire 0 = 1")
    expect(frEq(result.witness[1], a), "wire 1 = 42")
    expect(frEq(result.witness[2], b), "wire 2 = 99")
    expectEqual(result.evaluationOrder.count, 0, "no constraints to order")
}

// MARK: - Test 20: Single public input

private func testSinglePublicInput() {
    suite("Witness Solver: single public input")

    let engine = GPUR1CSWitnessSolverEngine()

    // Circuit: a * a = b  (squaring)
    // Wires: [1, a, b]  (a public => numPublic = 1)
    let c0 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 1, val: Fr.one)],
        b: [(col: 1, val: Fr.one)],
        c: [(col: 2, val: Fr.one)],
        index: 0
    )

    let system = GPUR1CSWitnessSolverEngine.makeSystem(
        constraints: [c0], numVars: 3, numPublic: 1
    )

    let a = frFromInt(12)
    let result = engine.solve(system: system, publicInputs: [a])

    expect(result.isComplete, "squaring circuit solved")
    expect(frEq(result.witness[2], frMul(a, a)), "wire 2 = a^2 = 144")
}

// MARK: - Test 21: Comparison hint (bit decomposition)

private func testComparisonHint() {
    suite("Witness Solver: comparison hint")

    let engine = GPUR1CSWitnessSolverEngine()

    // Simple: prove a value is a bit (0 or 1): b * (b - 1) = 0
    // Encoded as: w[1] * (w[1] - w[0]) = 0
    // Which is: w[1] * w[2] = 0, where w[2] = w[1] - 1
    // Constraint 0: (w[1] - w[0]) * 1 = w[2]  i.e. w[2] = w[1] - 1
    // Constraint 1: w[1] * w[2] = 0           i.e. w[1]*(w[1]-1) = 0
    // Wires: [1, bit, bit_minus_1]

    let c0 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 1, val: Fr.one), (col: 0, val: frSub(Fr.zero, Fr.one))],
        b: [(col: 0, val: Fr.one)],
        c: [(col: 2, val: Fr.one)],
        index: 0
    )

    // Constraint 1: w[1] * w[2] = 0 (zero on C side means empty C)
    // Actually: w[1] * w[2] = 0 means C side is zero linear combination
    let c1 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 1, val: Fr.one)],
        b: [(col: 2, val: Fr.one)],
        c: [],
        index: 1
    )

    let system = GPUR1CSWitnessSolverEngine.makeSystem(
        constraints: [c0, c1], numVars: 3, numPublic: 1
    )

    // Test with bit = 1
    let result = engine.solve(system: system, publicInputs: [Fr.one])
    expect(result.isComplete, "bit=1 circuit solved")
    // w[2] = 1 - 1 = 0
    expect(frEq(result.witness[2], Fr.zero), "bit_minus_1 = 0")

    // Validate
    let validation = engine.validate(system: system, witness: result.witness)
    expect(validation.valid, "bit=1 witness validates")

    // Test with bit = 0
    let result0 = engine.solve(system: system, publicInputs: [Fr.zero])
    expect(result0.isComplete, "bit=0 circuit solved")
    // w[2] = 0 - 1 = -1 mod p
    let negOne = frSub(Fr.zero, Fr.one)
    expect(frEq(result0.witness[2], negOne), "bit_minus_1 = -1")

    let validation0 = engine.validate(system: system, witness: result0.witness)
    expect(validation0.valid, "bit=0 witness validates")
}

// MARK: - Test 22: Incremental multi-stage

private func testIncrementalMultiStage() {
    suite("Witness Solver: incremental multi-stage")

    let engine = GPUR1CSWitnessSolverEngine()

    // 3-stage incremental build:
    // Stage 1: a * b = c
    // Stage 2: c + a = d  (encoded as (c + a) * 1 = d)
    // Stage 3: d * b = e
    // Wires: [1, a, b, c, d, e]
    let a = frFromInt(2)
    let b = frFromInt(5)

    var state = engine.makeIncrementalState(numVars: 6, numPublic: 2,
                                             publicInputs: [a, b])

    // Stage 1
    let s1 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 1, val: Fr.one)],
        b: [(col: 2, val: Fr.one)],
        c: [(col: 3, val: Fr.one)],
        index: 0
    )
    _ = engine.solveIncremental(state: &state, newConstraints: [s1])
    let c = frMul(a, b) // 10
    expect(frEq(state.witness[3], c), "stage 1: c = a*b = 10")

    // Stage 2
    let s2 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 3, val: Fr.one), (col: 1, val: Fr.one)],
        b: [(col: 0, val: Fr.one)],
        c: [(col: 4, val: Fr.one)],
        index: 1
    )
    _ = engine.solveIncremental(state: &state, newConstraints: [s2])
    let d = frAdd(c, a) // 12
    expect(frEq(state.witness[4], d), "stage 2: d = c+a = 12")

    // Stage 3
    let s3 = GPUR1CSWitnessSolverEngine.makeConstraint(
        a: [(col: 4, val: Fr.one)],
        b: [(col: 2, val: Fr.one)],
        c: [(col: 5, val: Fr.one)],
        index: 2
    )
    _ = engine.solveIncremental(state: &state, newConstraints: [s3])
    let e = frMul(d, b) // 60
    expect(frEq(state.witness[5], e), "stage 3: e = d*b = 60")

    // Final validation: build full system
    let fullSystem = GPUR1CSWitnessSolverEngine.makeSystem(
        constraints: [s1, s2, s3], numVars: 6, numPublic: 2
    )
    let validation = engine.validate(system: fullSystem, witness: state.witness)
    expect(validation.valid, "full incremental witness validates")
}
