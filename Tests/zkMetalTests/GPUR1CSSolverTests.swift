// GPU R1CS Solver Engine tests
import zkMetal
import Foundation

private func frEqual(_ a: Fr, _ b: Fr) -> Bool {
    return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
           a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

public func runGPUR1CSSolverTests() {
    testSimpleMultiplication()
    testAdditionCircuit()
    testConstraintSatisfaction()
    testUnsatisfiedDetection()
    testMultiConstraintSystem()
    testWitnessAugmentation()
    testResiduals()
    testMakeConstraintHelper()
}

// MARK: - Test 1: Simple multiplication constraint

private func testSimpleMultiplication() {
    suite("GPU R1CS Solver: simple multiplication")

    // Circuit: a * b = c
    // Wires: [1, a, b, c]  (a, b public => numPublic = 2)
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    // Constraint: w[1] * w[2] = w[3]
    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    bE.append(R1CSEntry(row: 0, col: 2, val: one))
    cE.append(R1CSEntry(row: 0, col: 3, val: one))

    let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 2,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    let a = frFromInt(3)
    let b = frFromInt(7)
    let c = frMul(a, b)  // 21
    let witness = [Fr.one, a, b, c]

    let engine = GPUR1CSSolverEngine()
    let result = engine.checkSatisfaction(r1cs: r1cs, witness: witness)

    expect(result.satisfied, "a*b=c satisfied")
    expect(result.unsatisfiedIndices.isEmpty, "no unsatisfied constraints")
    expect(result.numConstraints == 1, "1 constraint")
    expect(frEqual(result.residuals[0], Fr.zero), "residual is zero")
}

// MARK: - Test 2: Addition circuit

private func testAdditionCircuit() {
    suite("GPU R1CS Solver: addition circuit")

    // Circuit: a + b = c  encoded as (a + b) * 1 = c
    // Wires: [1, a, b, c]  (a, b public => numPublic = 2)
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    // A side: w[1] + w[2]
    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    aE.append(R1CSEntry(row: 0, col: 2, val: one))
    // B side: 1 (the constant wire)
    bE.append(R1CSEntry(row: 0, col: 0, val: one))
    // C side: w[3]
    cE.append(R1CSEntry(row: 0, col: 3, val: one))

    let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 2,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    let a = frFromInt(10)
    let b = frFromInt(20)
    let c = frAdd(a, b)  // 30
    let witness = [Fr.one, a, b, c]

    let engine = GPUR1CSSolverEngine()
    let result = engine.checkSatisfaction(r1cs: r1cs, witness: witness)

    expect(result.satisfied, "a+b=c satisfied")
    expect(result.numConstraints == 1, "1 constraint")

    // Also test with wrong witness
    let badWitness = [Fr.one, a, b, frFromInt(99)]
    let badResult = engine.checkSatisfaction(r1cs: r1cs, witness: badWitness)
    expect(!badResult.satisfied, "wrong witness detected")
}

// MARK: - Test 3: Constraint satisfaction check

private func testConstraintSatisfaction() {
    suite("GPU R1CS Solver: constraint satisfaction")

    // Circuit: x * x = y  (squaring)
    // Wires: [1, x, y]  (x public => numPublic = 1)
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    bE.append(R1CSEntry(row: 0, col: 1, val: one))
    cE.append(R1CSEntry(row: 0, col: 2, val: one))

    let r1cs = R1CSInstance(numConstraints: 1, numVars: 3, numPublic: 1,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    let x = frFromInt(5)
    let y = frMul(x, x)  // 25
    let witness = [Fr.one, x, y]

    let engine = GPUR1CSSolverEngine()
    let result = engine.checkSatisfaction(r1cs: r1cs, witness: witness)

    expect(result.satisfied, "x^2=y satisfied for x=5, y=25")

    // Check CPU sparseMatVec directly
    let az = engine.cpuSparseMatVec(r1cs.aEntries, witness, numRows: 1)
    let bz = engine.cpuSparseMatVec(r1cs.bEntries, witness, numRows: 1)
    let cz = engine.cpuSparseMatVec(r1cs.cEntries, witness, numRows: 1)

    expect(frEqual(az[0], x), "A*w = x")
    expect(frEqual(bz[0], x), "B*w = x")
    expect(frEqual(cz[0], y), "C*w = y")
    expect(frEqual(frMul(az[0], bz[0]), cz[0]), "A*w . B*w == C*w")
}

// MARK: - Test 4: Unsatisfied detection

private func testUnsatisfiedDetection() {
    suite("GPU R1CS Solver: unsatisfied detection")

    // Circuit: a * b = c
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    bE.append(R1CSEntry(row: 0, col: 2, val: one))
    cE.append(R1CSEntry(row: 0, col: 3, val: one))

    let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 2,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    // Wrong witness: 3 * 7 != 42
    let a = frFromInt(3)
    let b = frFromInt(7)
    let wrongC = frFromInt(42)  // should be 21
    let witness = [Fr.one, a, b, wrongC]

    let engine = GPUR1CSSolverEngine()
    let result = engine.checkSatisfaction(r1cs: r1cs, witness: witness)

    expect(!result.satisfied, "wrong witness detected as unsatisfied")
    expect(result.unsatisfiedIndices.count == 1, "exactly 1 unsatisfied constraint")
    expect(result.unsatisfiedIndices[0] == 0, "constraint 0 is unsatisfied")
    expect(!frEqual(result.residuals[0], Fr.zero), "residual is nonzero")

    // Verify residual = a*b - wrongC = 21 - 42 = -21 (mod p)
    let expectedResidual = frSub(frMul(a, b), wrongC)
    expect(frEqual(result.residuals[0], expectedResidual), "residual = a*b - wrongC")
}

// MARK: - Test 5: Multi-constraint system

private func testMultiConstraintSystem() {
    suite("GPU R1CS Solver: multi-constraint")

    // Circuit: a*b=c, c*d=e
    // Wires: [1, a, b, d, c, e]  (a, b, d public => numPublic = 3)
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    // Constraint 0: w[1] * w[2] = w[4]
    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    bE.append(R1CSEntry(row: 0, col: 2, val: one))
    cE.append(R1CSEntry(row: 0, col: 4, val: one))

    // Constraint 1: w[4] * w[3] = w[5]
    aE.append(R1CSEntry(row: 1, col: 4, val: one))
    bE.append(R1CSEntry(row: 1, col: 3, val: one))
    cE.append(R1CSEntry(row: 1, col: 5, val: one))

    let r1cs = R1CSInstance(numConstraints: 2, numVars: 6, numPublic: 3,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    let a = frFromInt(2)
    let b = frFromInt(5)
    let d = frFromInt(4)
    let c_val = frMul(a, b)        // 10
    let e_val = frMul(c_val, d)    // 40

    let witness = [Fr.one, a, b, d, c_val, e_val]

    let engine = GPUR1CSSolverEngine()
    let result = engine.checkSatisfaction(r1cs: r1cs, witness: witness)

    expect(result.satisfied, "chain circuit a*b=c, c*d=e satisfied")
    expect(result.numConstraints == 2, "2 constraints")
    expect(result.unsatisfiedIndices.isEmpty, "no unsatisfied")

    // Test with wrong e
    let badWitness = [Fr.one, a, b, d, c_val, frFromInt(99)]
    let badResult = engine.checkSatisfaction(r1cs: r1cs, witness: badWitness)

    expect(!badResult.satisfied, "chain with wrong e detected")
    expect(badResult.unsatisfiedIndices.count == 1, "1 unsatisfied constraint")
    expect(badResult.unsatisfiedIndices[0] == 1, "constraint 1 is bad")

    // Both wrong
    let veryBadWitness = [Fr.one, a, b, d, frFromInt(99), frFromInt(99)]
    let veryBadResult = engine.checkSatisfaction(r1cs: r1cs, witness: veryBadWitness)
    expect(!veryBadResult.satisfied, "both wrong detected")
    expect(veryBadResult.unsatisfiedIndices.count == 2, "both constraints unsatisfied")
}

// MARK: - Test 6: Witness augmentation

private func testWitnessAugmentation() {
    suite("GPU R1CS Solver: witness augmentation")

    // Circuit: a * b = c
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    bE.append(R1CSEntry(row: 0, col: 2, val: one))
    cE.append(R1CSEntry(row: 0, col: 3, val: one))

    let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 2,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    let a = frFromInt(6)
    let b = frFromInt(9)

    let engine = GPUR1CSSolverEngine()
    let augResult = engine.augmentWitness(r1cs: r1cs, publicInputs: [a, b])

    expect(augResult.isComplete, "witness fully augmented")
    expect(frEqual(augResult.witness[0], Fr.one), "wire 0 = 1")
    expect(frEqual(augResult.witness[1], a), "wire 1 = a")
    expect(frEqual(augResult.witness[2], b), "wire 2 = b")
    expect(frEqual(augResult.witness[3], frMul(a, b)), "wire 3 = a*b = 54")

    // Verify the augmented witness satisfies the R1CS
    let satResult = engine.checkSatisfaction(r1cs: r1cs, witness: augResult.witness)
    expect(satResult.satisfied, "augmented witness satisfies R1CS")
}

// MARK: - Test 7: Residuals

private func testResiduals() {
    suite("GPU R1CS Solver: residuals")

    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    bE.append(R1CSEntry(row: 0, col: 2, val: one))
    cE.append(R1CSEntry(row: 0, col: 3, val: one))

    let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 2,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    let a = frFromInt(4)
    let b = frFromInt(5)
    let c = frMul(a, b)

    let engine = GPUR1CSSolverEngine()

    // Correct witness: residual should be zero
    let res = engine.residuals(r1cs: r1cs, witness: [Fr.one, a, b, c])
    expect(res.count == 1, "1 residual")
    expect(frEqual(res[0], Fr.zero), "correct witness has zero residual")

    // Wrong witness: residual should be nonzero
    let badRes = engine.residuals(r1cs: r1cs, witness: [Fr.one, a, b, frFromInt(99)])
    expect(!frEqual(badRes[0], Fr.zero), "wrong witness has nonzero residual")
}

// MARK: - Test 8: makeConstraint helper

private func testMakeConstraintHelper() {
    suite("GPU R1CS Solver: makeConstraint helper")

    let one = Fr.one
    let two = frFromInt(2)

    // Build constraint: 2*w[1] * w[2] = w[3]
    let (aE, bE, cE) = GPUR1CSSolverEngine.makeConstraint(
        a: [(col: 1, val: two)],
        b: [(col: 2, val: one)],
        c: [(col: 3, val: one)],
        row: 0
    )

    let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 2,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    // 2*3 * 7 = 42
    let witness = [Fr.one, frFromInt(3), frFromInt(7), frFromInt(42)]
    let engine = GPUR1CSSolverEngine()
    let result = engine.checkSatisfaction(r1cs: r1cs, witness: witness)

    expect(result.satisfied, "2*a * b = c satisfied for a=3, b=7, c=42")
}
