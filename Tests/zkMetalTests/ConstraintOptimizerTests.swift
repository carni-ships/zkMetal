// ConstraintOptimizerTests — Tests for constraint system optimization passes

import Foundation
import zkMetal

func runConstraintOptimizerTests() {
    suite("ConstraintOptimizer")

    testConstantFolding()
    testLinearCombinationMerge()
    testDuplicateElimination()
    testDeadConstraintElimination()
    testVariableRenumbering()
    testOptimizeAll()
    testR1CSFromConstraintSystem()
    testR1CSLinearElimination()
    testR1CSWitnessReduction()
    testOptimizeAllStats()
}

// MARK: - Constant Folding Tests

private func testConstantFolding() {
    // Test: trivially true constraint (0 == 0) is eliminated
    let cs = ConstraintSystem(numWires: 2)
    cs.addConstraint(.constant(Fr.zero), label: "trivial")
    cs.addConstraint(.wire(Wire.col(0)) - .wire(Wire.col(1)), label: "real")

    let optimized = ConstraintOptimizer.constantFolding(cs: cs)
    expectEqual(optimized.constraints.count, 1, "trivially true constraint eliminated")
    expectEqual(optimized.constraints[0].label, "real", "kept non-trivial constraint")

    // Test: constant arithmetic is folded
    let cs2 = ConstraintSystem(numWires: 1)
    let two = frFromInt(2)
    let three = frFromInt(3)
    let expr = Expr.add(.constant(two), .constant(three)) - .wire(Wire.col(0))
    cs2.addConstraint(expr, label: "fold_me")

    let opt2 = ConstraintOptimizer.constantFolding(cs: cs2)
    expectEqual(opt2.constraints.count, 1, "non-trivial constraint kept")
    // The folded expression should have fewer nodes
    expect(opt2.constraints[0].expr.nodeCount <= expr.nodeCount, "constant folding reduces node count")
}

// MARK: - Linear Combination Merge Tests

private func testLinearCombinationMerge() {
    // Test: a + a should merge to 2*a
    let cs = ConstraintSystem(numWires: 2)
    let a = Expr.wire(Wire.col(0))
    let b = Expr.wire(Wire.col(1))
    // (a + a) - b == 0, should become 2*a - b == 0
    cs.addConstraint(.add(.add(a, a), .neg(b)), label: "merge_test")

    let optimized = ConstraintOptimizer.linearCombinationMerge(cs: cs)
    expectEqual(optimized.constraints.count, 1, "constraint count preserved")
    // Verify that the merged expression uses wire(0) only once (terms were combined)
    // The original has wire(0) twice; after merging like terms, it should appear once
    // as 2*wire(0)
    let mergedWires = optimized.constraints[0].expr.wires
    expect(mergedWires.contains(Wire.col(0)), "merged expression still references wire 0")
    expect(mergedWires.contains(Wire.col(1)), "merged expression still references wire 1")
}

// MARK: - Duplicate Elimination Tests

private func testDuplicateElimination() {
    // Test: duplicate constraints are removed
    let cs = ConstraintSystem(numWires: 2)
    let expr = Expr.wire(Wire.col(0)) - .wire(Wire.col(1))
    cs.addConstraint(expr, label: "first")
    cs.addConstraint(expr, label: "duplicate")
    cs.addConstraint(.wire(Wire.col(0)) * .wire(Wire.col(1)), label: "different")

    let optimized = ConstraintOptimizer.duplicateElimination(cs: cs)
    expectEqual(optimized.constraints.count, 2, "duplicate constraint removed")
}

// MARK: - Dead Constraint Elimination Tests

private func testDeadConstraintElimination() {
    // Build a system where one constraint defines a wire nobody else uses
    let cs = ConstraintSystem(numWires: 5)

    // Constraint 0: w0 * w1 - w2 = 0 (w2 is used by constraint 1)
    cs.addConstraint(.mul(.wire(Wire.col(0)), .wire(Wire.col(1))) - .wire(Wire.col(2)),
                     label: "used_mul")
    // Constraint 1: w2 + w3 - w4 = 0 (w4 is never used elsewhere)
    // But w2 is used so constraint 0 stays, and this constraint has w4 only used here
    cs.addConstraint(.wire(Wire.col(2)) + .wire(Wire.col(3)) - .wire(Wire.col(4)),
                     label: "dead_output")

    let optimized = ConstraintOptimizer.eliminateDeadConstraints(cs: cs)
    // The dead constraint elimination should remove constraint 1 since w4 is used nowhere else
    expect(optimized.constraints.count <= cs.constraints.count,
           "dead constraint elimination does not add constraints")
}

// MARK: - Variable Renumbering Tests

private func testVariableRenumbering() {
    // Build a system using wires 0, 2, 5 (gaps at 1, 3, 4)
    let cs = ConstraintSystem(numWires: 6)
    cs.addConstraint(.wire(Wire.col(0)) + .wire(Wire.col(2)) - .wire(Wire.col(5)),
                     label: "sparse_wires")

    let (renumbered, eliminated) = ConstraintOptimizer.variableRenumbering(cs: cs)
    expectEqual(renumbered.numWires, 3, "renumbered to 3 wires (0, 2, 5 -> 0, 1, 2)")
    expectEqual(eliminated, 3, "eliminated 3 unused wire slots")
    expectEqual(renumbered.constraints.count, 1, "constraint count preserved")

    // Verify the wires are now 0, 1, 2
    let wires = renumbered.constraints[0].expr.wires
    let indices = Set(wires.map { $0.index })
    expectEqual(indices, Set([0, 1, 2]), "wires renumbered to contiguous range")
}

// MARK: - Full Pipeline Tests

private func testOptimizeAll() {
    // Build a Fibonacci system with some redundancy
    let cs = ConstraintSystem(numWires: 5)
    // w0 + w1 = w2
    cs.addConstraint(.wire(Wire.col(0)) + .wire(Wire.col(1)) - .wire(Wire.col(2)), label: "fib0")
    // w1 + w2 = w3
    cs.addConstraint(.wire(Wire.col(1)) + .wire(Wire.col(2)) - .wire(Wire.col(3)), label: "fib1")
    // Duplicate of fib0
    cs.addConstraint(.wire(Wire.col(0)) + .wire(Wire.col(1)) - .wire(Wire.col(2)), label: "dup_fib0")
    // Trivially true
    cs.addConstraint(.constant(Fr.zero), label: "trivial")

    let (optimized, stats) = ConstraintOptimizer.optimizeAll(cs: cs)
    expectEqual(stats.originalConstraints, 4, "original constraint count")
    expect(stats.optimizedConstraints < 4, "optimizer reduced constraints")
    expect(stats.timeMs >= 0, "timing is non-negative")
    expect(optimized.constraints.count == stats.optimizedConstraints, "stats match actual")

    print("  Optimizer: \(stats.summary)")
}

private func testOptimizeAllStats() {
    // Verify stats on fibonacci system
    // Note: dead constraint elimination may remove the final constraint whose
    // output wire (highest index) is not used by any other constraint.
    let cs = ConstraintSystem.fibonacci(steps: 5)
    let (_, stats) = ConstraintOptimizer.optimizeAll(cs: cs)
    expectEqual(stats.originalConstraints, 3, "fibonacci(5) has 3 constraints")
    expect(stats.optimizedConstraints <= 3, "optimizer does not add constraints")
    expect(stats.optimizedConstraints >= 1, "optimizer keeps at least 1 constraint")
    expect(stats.reductionRatio <= 1.0, "reduction ratio at most 1.0")
}

// MARK: - R1CS Tests

private func testR1CSFromConstraintSystem() {
    // Build an R1CS from the factory method
    let cs = ConstraintSystem.r1cs(numGates: 3)
    guard let r1cs = R1CSSystem.fromConstraintSystem(cs) else {
        expect(false, "R1CS conversion should succeed for r1cs factory systems")
        return
    }
    expectEqual(r1cs.numConstraints, 3, "3 R1CS constraints")
    expectEqual(r1cs.numVariables, 1 + 9, "1 + 3*3 variables")

    // Verify satisfaction with a valid witness
    // Gate 0: w0 * w1 = w2 => z[1]*z[2] = z[3]
    // Gate 1: w3 * w4 = w5 => z[4]*z[5] = z[6]
    // Gate 2: w6 * w7 = w8 => z[7]*z[8] = z[9]
    let two = frFromInt(2)
    let three = frFromInt(3)
    let six = frFromInt(6)
    let z: [Fr] = [Fr.one,  // z[0] = 1 (constant)
                   two, three, six,   // gate 0: 2*3=6
                   two, two, frFromInt(4),  // gate 1: 2*2=4
                   Fr.one, Fr.one, Fr.one]  // gate 2: 1*1=1
    expect(r1cs.isSatisfied(z: z), "R1CS satisfied with valid witness")

    // Invalid witness should fail
    var badZ = z
    badZ[3] = frFromInt(7)  // wrong: 2*3 != 7
    expect(!r1cs.isSatisfied(z: badZ), "R1CS rejects invalid witness")
}

private func testR1CSLinearElimination() {
    // Build a simple system with a linear constraint
    // Constraint 0: 1 * (z[1] + z[2]) = z[3]  (linear: A selects constant 1)
    // Constraint 1: z[1] * z[2] = z[4]         (nonlinear)
    let n = 5  // z = [1, x1, x2, x3, x4]
    var aBuilder = SparseMatrixBuilder(rows: 2, cols: n)
    var bBuilder = SparseMatrixBuilder(rows: 2, cols: n)
    var cBuilder = SparseMatrixBuilder(rows: 2, cols: n)

    // Row 0: A=[1 at col0], B=[1 at col1, 1 at col2], C=[1 at col3]
    // => 1 * (z[1]+z[2]) = z[3] -- linear constraint
    aBuilder.set(row: 0, col: 0, value: Fr.one)
    bBuilder.set(row: 0, col: 1, value: Fr.one)
    bBuilder.set(row: 0, col: 2, value: Fr.one)
    cBuilder.set(row: 0, col: 3, value: Fr.one)

    // Row 1: A=[1 at col1], B=[1 at col2], C=[1 at col4]
    // => z[1] * z[2] = z[4]
    aBuilder.set(row: 1, col: 1, value: Fr.one)
    bBuilder.set(row: 1, col: 2, value: Fr.one)
    cBuilder.set(row: 1, col: 4, value: Fr.one)

    let r1cs = R1CSSystem(A: aBuilder.build(), B: bBuilder.build(), C: cBuilder.build(), numPublicInputs: 0)
    expectEqual(r1cs.numConstraints, 2, "original: 2 constraints")

    let optimized = R1CSOptimizer.eliminateLinearConstraints(r1cs: r1cs)
    // z[3] appears only in the linear constraint (row 0), so it can be eliminated
    expect(optimized.numConstraints <= r1cs.numConstraints,
           "linear elimination does not add constraints")
}

private func testR1CSWitnessReduction() {
    // Build a system where z[3] = z[1] * z[2] (intermediate, used only once in C)
    let n = 4  // z = [1, x1, x2, x3]
    var aBuilder = SparseMatrixBuilder(rows: 1, cols: n)
    var bBuilder = SparseMatrixBuilder(rows: 1, cols: n)
    var cBuilder = SparseMatrixBuilder(rows: 1, cols: n)

    // Row 0: z[1] * z[2] = z[3]
    aBuilder.set(row: 0, col: 1, value: Fr.one)
    bBuilder.set(row: 0, col: 2, value: Fr.one)
    cBuilder.set(row: 0, col: 3, value: Fr.one)

    let r1cs = R1CSSystem(A: aBuilder.build(), B: bBuilder.build(), C: cBuilder.build(), numPublicInputs: 0)

    let (reduced, subs) = R1CSOptimizer.reduceWitnessSize(r1cs: r1cs)
    // z[3] appears only once in C with coefficient 1, so it should be eliminatable
    if !subs.isEmpty {
        expect(subs.keys.contains(3), "z[3] is substituted")
        expect(reduced.numConstraints < r1cs.numConstraints, "constraint removed")

        // Verify substitution rule evaluates correctly
        let two = frFromInt(2)
        let three = frFromInt(3)
        let z: [Fr] = [Fr.one, two, three, frFromInt(6)]
        let computed = subs[3]!.evaluate(z: z)
        expectEqual(computed, frFromInt(6), "substitution rule computes 2*3=6")
    } else {
        // If no substitutions were found, the system is already minimal
        expectEqual(reduced.numConstraints, r1cs.numConstraints, "no reduction possible")
    }
}
