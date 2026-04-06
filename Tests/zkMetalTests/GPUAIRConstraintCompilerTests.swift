// GPU AIR Constraint Compiler Tests — validates the Fr-based compiler,
// expression compilation, degree analysis, composition, batch evaluation,
// constant folding, standard library AIRs, and error handling.

import Foundation
@testable import zkMetal

public func runGPUAIRConstraintCompilerTests() {
    suite("GPUAIRCompiler - Expression Builder")
    testFrExprBuilderCol()
    testFrExprBuilderNext()
    testFrExprBuilderConstant()
    testFrExprBuilderArithmetic()

    suite("GPUAIRCompiler - Expression Compilation")
    testFrCompileSimple()
    testFrCompileTransition()
    testFrCompilePublicInput()

    suite("GPUAIRCompiler - Degree Analysis")
    testFrDegreeLinear()
    testFrDegreeQuadratic()
    testFrDegreeMixed()

    suite("GPUAIRCompiler - Composition")
    testFrComposeSingle()
    testFrComposeMultiple()

    suite("GPUAIRCompiler - Batch Evaluation")
    testFrBatchEval()
    testFrBatchEvalComposed()

    suite("GPUAIRCompiler - Constant Folding")
    testFrConstFoldAdd()
    testFrConstFoldMul()
    testFrConstFoldIdentities()
    testFrConstFoldPow()

    suite("GPUAIRCompiler - Fibonacci AIR")
    testFrFibonacciTrace()
    testFrFibonacciVerify()

    suite("GPUAIRCompiler - Range Check AIR")
    testFrRangeCheckTrace()
    testFrRangeCheckDegree()

    suite("GPUAIRCompiler - Permutation Check AIR")
    testFrPermutationTrace()
    testFrPermutationBoundary()

    suite("GPUAIRCompiler - Error Handling")
    testFrErrorNoColumns()
    testFrErrorNoConstraints()
    testFrErrorColumnOutOfRange()
    testFrErrorUnknownPublicInput()
}

// MARK: - Expression Builder

private func testFrExprBuilderCol() {
    let expr = FrAIRExprBuilder.col(0)
    if case .column(0) = expr {
        expect(true, "col(0) produces .column(0)")
    } else {
        expect(false, "col(0) should produce .column(0)")
    }

    let expr2 = FrAIRExprBuilder.col(3)
    if case .column(3) = expr2 {
        expect(true, "col(3) produces .column(3)")
    } else {
        expect(false, "col(3) should produce .column(3)")
    }
}

private func testFrExprBuilderNext() {
    let expr = FrAIRExprBuilder.next(1)
    if case .nextColumn(1) = expr {
        expect(true, "next(1) produces .nextColumn(1)")
    } else {
        expect(false, "next(1) should produce .nextColumn(1)")
    }
}

private func testFrExprBuilderConstant() {
    let expr = FrAIRExprBuilder.constant(42)
    if case .constant(let v) = expr {
        let val = frToUInt64(v)
        expectEqual(val, UInt64(42), "constant(42) value")
    } else {
        expect(false, "constant(42) should produce .constant")
    }

    let expr2 = FrAIRExprBuilder.constant(Fr.one)
    if case .constant(let v) = expr2 {
        expect(v == Fr.one, "constant(Fr.one) = Fr.one")
    } else {
        expect(false, "constant(Fr.one) should produce .constant")
    }
}

private func testFrExprBuilderArithmetic() {
    let a = FrAIRExprBuilder.col(0)
    let b = FrAIRExprBuilder.col(1)

    let sum = a + b
    expectEqual(sum.degree, 1, "col(0) + col(1) degree = 1")

    let diff = FrAIRExprBuilder.next(0) - b
    expectEqual(diff.degree, 1, "next(0) - col(1) degree = 1")
    expect(diff.usesNextRow, "next(0) - col(1) uses next row")

    let prod = a * b
    expectEqual(prod.degree, 2, "col(0) * col(1) degree = 2")

    let cubed = FrAIRExpression.pow(a, 3)
    expectEqual(cubed.degree, 3, "col(0)^3 degree = 3")
}

// MARK: - Expression Compilation

private func testFrCompileSimple() {
    // col(0) - col(1)
    let expr = FrAIRExprBuilder.col(0) - FrAIRExprBuilder.col(1)
    do {
        let compiled = try GPUAIRConstraintCompiler.compileExpression(expr, numColumns: 2)
        let five = frFromInt(5)
        let three = frFromInt(3)
        let result = compiled.evaluate([five, three], [Fr.zero, Fr.zero])
        let expected = frFromInt(2)
        expect(result == expected, "5 - 3 = 2")

        let ten = frFromInt(10)
        let result2 = compiled.evaluate([ten, ten], [Fr.zero, Fr.zero])
        expect(result2.isZero, "10 - 10 = 0")
    } catch {
        expect(false, "Compile failed: \(error)")
    }
}

private func testFrCompileTransition() {
    // next(0) - (col(0) + col(1)) => Fibonacci transition
    let expr = FrAIRExprBuilder.next(0) - (FrAIRExprBuilder.col(0) + FrAIRExprBuilder.col(1))
    do {
        let compiled = try GPUAIRConstraintCompiler.compileExpression(expr, numColumns: 2)

        // Valid: current=[1,1], next=[2,x] => next(0)-(1+1) = 2-2 = 0
        let one = Fr.one
        let two = frFromInt(2)
        let result = compiled.evaluate([one, one], [two, Fr.zero])
        expect(result.isZero, "Valid Fibonacci step: 2 - (1+1) = 0")

        // Invalid: current=[1,1], next=[3,x] => next(0)-(1+1) = 3-2 = 1
        let three = frFromInt(3)
        let result2 = compiled.evaluate([one, one], [three, Fr.zero])
        expect(!result2.isZero, "Invalid Fibonacci step: 3 - (1+1) != 0")
    } catch {
        expect(false, "Compile transition failed: \(error)")
    }
}

private func testFrCompilePublicInput() {
    // col(0) - pub("init")
    let expr = FrAIRExprBuilder.col(0) - FrAIRExprBuilder.publicInput("init")
    do {
        let initVal = frFromInt(42)
        let compiled = try GPUAIRConstraintCompiler.compileExpression(
            expr, numColumns: 1, publicInputValues: ["init": initVal])

        let result = compiled.evaluate([initVal], [Fr.zero])
        expect(result.isZero, "col(0) = pub(init) => 0")

        let ten = frFromInt(10)
        let result2 = compiled.evaluate([ten], [Fr.zero])
        expect(!result2.isZero, "col(0) != pub(init) => nonzero")
    } catch {
        expect(false, "Compile public input failed: \(error)")
    }
}

// MARK: - Degree Analysis

private func testFrDegreeLinear() {
    let analysis = GPUAIRConstraintCompiler.analyzeDegrees(
        [FrAIRExprBuilder.next(0) - FrAIRExprBuilder.col(1)],
        logTraceLength: 4)
    expectEqual(analysis.transitionDegrees, [1], "Linear degree = [1]")
    expectEqual(analysis.maxTransitionDegree, 1, "Max degree = 1")
    expectEqual(analysis.compositionDegreeBound, 16, "Bound = 1 * 16")
    expectEqual(analysis.numQuotientChunks, 1, "1 chunk for linear")
}

private func testFrDegreeQuadratic() {
    let diff = FrAIRExprBuilder.next(0) - FrAIRExprBuilder.col(0)
    let constraint = diff * (FrAIRExprBuilder.constant(65536) - diff)
    let analysis = GPUAIRConstraintCompiler.analyzeDegrees([constraint], logTraceLength: 3)
    expectEqual(analysis.maxTransitionDegree, 2, "Quadratic max degree = 2")
    expectEqual(analysis.numQuotientChunks, 2, "2 chunks for quadratic")
}

private func testFrDegreeMixed() {
    let constraints: [FrAIRExpression] = [
        FrAIRExprBuilder.next(0) - FrAIRExprBuilder.col(1),
        FrAIRExprBuilder.col(0) * FrAIRExprBuilder.col(1),
        .pow(FrAIRExprBuilder.col(0), 3)
    ]
    let analysis = GPUAIRConstraintCompiler.analyzeDegrees(constraints, logTraceLength: 4)
    expectEqual(analysis.transitionDegrees, [1, 2, 3], "Mixed degrees")
    expectEqual(analysis.maxTransitionDegree, 3, "Max of mixed = 3")
}

// MARK: - Composition

private func testFrComposeSingle() {
    do {
        let expr = FrAIRExprBuilder.col(0) - FrAIRExprBuilder.constant(5)
        let compiled = try GPUAIRConstraintCompiler.compileExpression(expr, numColumns: 1)
        let alpha = frFromInt(7)
        let composed = GPUAIRConstraintCompiler.composeConstraints([compiled], alpha: alpha)

        let five = frFromInt(5)
        let result = composed([five], [Fr.zero])
        expect(result.isZero, "Satisfied single constraint => 0")

        let three = frFromInt(3)
        let result2 = composed([three], [Fr.zero])
        expect(!result2.isZero, "Violated single constraint => nonzero")
    } catch {
        expect(false, "Single composition failed: \(error)")
    }
}

private func testFrComposeMultiple() {
    do {
        // C0: col(0) - 1 = 0
        // C1: col(1) - 2 = 0
        let c0 = try GPUAIRConstraintCompiler.compileExpression(
            FrAIRExprBuilder.col(0) - FrAIRExprBuilder.constant(1), numColumns: 2)
        let c1 = try GPUAIRConstraintCompiler.compileExpression(
            FrAIRExprBuilder.col(1) - FrAIRExprBuilder.constant(2), numColumns: 2)

        let alpha = frFromInt(3)
        let composed = GPUAIRConstraintCompiler.composeConstraints([c0, c1], alpha: alpha)

        // Both satisfied
        let one = Fr.one
        let two = frFromInt(2)
        let result = composed([one, two], [Fr.zero, Fr.zero])
        expect(result.isZero, "Both satisfied => 0")

        // C0 violated: col(0)=5 => C0=4, C1=0 => 1*4 + 3*0 = 4
        let five = frFromInt(5)
        let result2 = composed([five, two], [Fr.zero, Fr.zero])
        let expected2 = frFromInt(4)
        expect(result2 == expected2, "C0=4, C1=0 => 4")

        // C1 violated: col(0)=1, col(1)=5 => C0=0, C1=3 => 1*0 + 3*3 = 9
        let result3 = composed([one, five], [Fr.zero, Fr.zero])
        let expected3 = frFromInt(9)
        expect(result3 == expected3, "C0=0, C1=3, alpha=3 => 9")
    } catch {
        expect(false, "Multi composition failed: \(error)")
    }
}

// MARK: - Batch Evaluation

private func testFrBatchEval() {
    do {
        // Simple constraint: next(0) - col(0) - const(1) = 0 (counter)
        let expr = FrAIRExprBuilder.next(0) - FrAIRExprBuilder.col(0) - FrAIRExprBuilder.constant(1)
        let compiled = try GPUAIRConstraintCompiler.compileExpression(expr, numColumns: 1)

        // Valid counting trace: 0, 1, 2, 3
        let trace: [[Fr]] = [[frFromInt(0), frFromInt(1), frFromInt(2), frFromInt(3)]]
        let results = GPUAIRConstraintCompiler.batchEvaluate(
            constraints: [compiled], trace: trace, numRows: 4)

        expectEqual(results.count, 1, "1 constraint")
        expectEqual(results[0].count, 3, "3 transition evaluations for 4 rows")
        for i in 0..<3 {
            expect(results[0][i].isZero, "Counter constraint satisfied at row \(i)")
        }
    } catch {
        expect(false, "Batch eval failed: \(error)")
    }
}

private func testFrBatchEvalComposed() {
    do {
        let expr = FrAIRExprBuilder.next(0) - FrAIRExprBuilder.col(0) - FrAIRExprBuilder.constant(1)
        let compiled = try GPUAIRConstraintCompiler.compileExpression(expr, numColumns: 1)

        let trace: [[Fr]] = [[frFromInt(0), frFromInt(1), frFromInt(2), frFromInt(3)]]
        let alpha = frFromInt(7)
        let results = GPUAIRConstraintCompiler.batchEvaluateComposed(
            constraints: [compiled], trace: trace, numRows: 4, alpha: alpha)

        expectEqual(results.count, 3, "3 rows of composed evaluation")
        for i in 0..<3 {
            expect(results[i].isZero, "Composed constraint satisfied at row \(i)")
        }
    } catch {
        expect(false, "Batch eval composed failed: \(error)")
    }
}

// MARK: - Constant Folding

private func testFrConstFoldAdd() {
    let three = frFromInt(3)
    let four = frFromInt(4)
    let expr = FrAIRExpression.add(.constant(three), .constant(four))
    let optimized = GPUAIRConstraintCompiler.optimizeExpression(expr)
    if case .constant(let v) = optimized {
        let val = frToUInt64(v)
        expectEqual(val, UInt64(7), "3 + 4 folded to 7")
    } else {
        expect(false, "Should fold to constant")
    }
}

private func testFrConstFoldMul() {
    let five = frFromInt(5)
    let six = frFromInt(6)
    let expr = FrAIRExpression.mul(.constant(five), .constant(six))
    let optimized = GPUAIRConstraintCompiler.optimizeExpression(expr)
    if case .constant(let v) = optimized {
        let val = frToUInt64(v)
        expectEqual(val, UInt64(30), "5 * 6 folded to 30")
    } else {
        expect(false, "Should fold to constant")
    }
}

private func testFrConstFoldIdentities() {
    let col = FrAIRExpression.column(0)

    // 0 + x = x
    let addZero = GPUAIRConstraintCompiler.optimizeExpression(
        .add(.constant(Fr.zero), col))
    if case .column(0) = addZero {
        expect(true, "0 + x = x")
    } else {
        expect(false, "0 + x should fold to x")
    }

    // x * 1 = x
    let mulOne = GPUAIRConstraintCompiler.optimizeExpression(
        .mul(col, .constant(Fr.one)))
    if case .column(0) = mulOne {
        expect(true, "x * 1 = x")
    } else {
        expect(false, "x * 1 should fold to x")
    }

    // x * 0 = 0
    let mulZero = GPUAIRConstraintCompiler.optimizeExpression(
        .mul(col, .constant(Fr.zero)))
    if case .constant(let v) = mulZero {
        expect(v.isZero, "x * 0 = 0")
    } else {
        expect(false, "x * 0 should fold to 0")
    }

    // x - 0 = x
    let subZero = GPUAIRConstraintCompiler.optimizeExpression(
        .sub(col, .constant(Fr.zero)))
    if case .column(0) = subZero {
        expect(true, "x - 0 = x")
    } else {
        expect(false, "x - 0 should fold to x")
    }
}

private func testFrConstFoldPow() {
    let two = frFromInt(2)
    let expr = FrAIRExpression.pow(.constant(two), 10)
    let optimized = GPUAIRConstraintCompiler.optimizeExpression(expr)
    if case .constant(let v) = optimized {
        let val = frToUInt64(v)
        expectEqual(val, UInt64(1024), "2^10 folded to 1024")
    } else {
        expect(false, "constant^n should fold")
    }

    // x^0 = 1
    let powZero = GPUAIRConstraintCompiler.optimizeExpression(
        .pow(.column(0), 0))
    if case .constant(let v) = powZero {
        expect(v == Fr.one, "x^0 = 1")
    } else {
        expect(false, "x^0 should fold to 1")
    }

    // x^1 = x
    let powOne = GPUAIRConstraintCompiler.optimizeExpression(
        .pow(.column(0), 1))
    if case .column(0) = powOne {
        expect(true, "x^1 = x")
    } else {
        expect(false, "x^1 should fold to x")
    }
}

// MARK: - Fibonacci AIR

private func testFrFibonacciTrace() {
    do {
        let air = try FrAIRStandardLibrary.fibonacci(logTraceLength: 4)
        let trace = air.generateTrace()

        expectEqual(trace.count, 2, "Fib: 2 columns")
        expectEqual(trace[0].count, 16, "Fib: 16 rows")

        // Initial values
        expect(trace[0][0] == Fr.one, "Fib a[0] = 1")
        expect(trace[1][0] == Fr.one, "Fib b[0] = 1")

        // Check Fibonacci relation
        for i in 0..<15 {
            expect(trace[0][i + 1] == trace[1][i], "Fib a[\(i+1)] = b[\(i)]")
            let sum = frAdd(trace[0][i], trace[1][i])
            expect(trace[1][i + 1] == sum, "Fib b[\(i+1)] = a[\(i)] + b[\(i)]")
        }
    } catch {
        expect(false, "Fibonacci build failed: \(error)")
    }
}

private func testFrFibonacciVerify() {
    do {
        let air = try FrAIRStandardLibrary.fibonacci(logTraceLength: 4)
        let trace = air.generateTrace()
        let err = air.verifyTrace(trace)
        expect(err == nil, "Fib trace valid: \(err ?? "")")
    } catch {
        expect(false, "Fibonacci verify failed: \(error)")
    }
}

// MARK: - Range Check AIR

private func testFrRangeCheckTrace() {
    do {
        let values: [Fr] = [frFromInt(100), frFromInt(50), frFromInt(200), frFromInt(10)]
        let air = try FrAIRStandardLibrary.rangeCheck(logTraceLength: 2, values: values)
        let trace = air.generateTrace()

        expectEqual(trace.count, 1, "RC: 1 column")
        expectEqual(trace[0].count, 4, "RC: 4 rows")

        // Should be sorted (non-decreasing)
        for i in 0..<3 {
            let a = frToUInt64(trace[0][i])
            let b = frToUInt64(trace[0][i + 1])
            expect(a <= b, "RC sorted: \(a) <= \(b)")
        }
    } catch {
        expect(false, "Range check build failed: \(error)")
    }
}

private func testFrRangeCheckDegree() {
    do {
        let values: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let air = try FrAIRStandardLibrary.rangeCheck(logTraceLength: 2, values: values)
        expectEqual(air.constraintDegrees.count, 1, "RC: 1 constraint")
        expectEqual(air.constraintDegrees[0], 2, "RC: degree 2 (quadratic)")
    } catch {
        expect(false, "Range check degree failed: \(error)")
    }
}

// MARK: - Permutation Check AIR

private func testFrPermutationTrace() {
    do {
        let valuesA: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let valuesB: [Fr] = [frFromInt(3), frFromInt(1), frFromInt(4), frFromInt(2)]
        let gamma = frFromInt(17)

        let air = try FrAIRStandardLibrary.permutationCheck(
            logTraceLength: 2, valuesA: valuesA, valuesB: valuesB, gamma: gamma)
        let trace = air.generateTrace()

        expectEqual(trace.count, 3, "Perm: 3 columns")
        expectEqual(trace[0].count, 4, "Perm: 4 rows")

        // Accumulator starts at 1
        expect(trace[2][0] == Fr.one, "Perm acc[0] = 1")

        // Values A match input
        for i in 0..<4 {
            expect(trace[0][i] == valuesA[i], "A[\(i)] matches")
        }
    } catch {
        expect(false, "Permutation build failed: \(error)")
    }
}

private func testFrPermutationBoundary() {
    do {
        let valuesA: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let valuesB: [Fr] = [frFromInt(20), frFromInt(40), frFromInt(10), frFromInt(30)]
        let gamma = frFromInt(5)

        let air = try FrAIRStandardLibrary.permutationCheck(
            logTraceLength: 2, valuesA: valuesA, valuesB: valuesB, gamma: gamma)

        let bcs = air.boundaryConstraints
        expect(bcs.count >= 1, "Perm has boundary constraints")
        let accBC = bcs.first { $0.column == 2 && $0.row == 0 }
        expect(accBC != nil, "Boundary for acc[0] exists")
        if let bc = accBC {
            expect(bc.value == Fr.one, "Boundary acc[0] = 1")
        }
    } catch {
        expect(false, "Permutation boundary failed: \(error)")
    }
}

// MARK: - Error Handling

private func testFrErrorNoColumns() {
    do {
        _ = try GPUAIRConstraintCompiler.compile(
            numColumns: 0,
            logTraceLength: 2,
            transitions: [FrAIRExprBuilder.col(0)],
            boundaries: [],
            traceGenerator: { [[]] })
        expect(false, "Should fail with 0 columns")
    } catch {
        expect(true, "Correctly rejected 0 columns")
    }
}

private func testFrErrorNoConstraints() {
    do {
        _ = try GPUAIRConstraintCompiler.compile(
            numColumns: 1,
            logTraceLength: 2,
            transitions: [],
            boundaries: [],
            traceGenerator: { [[Fr.zero]] })
        expect(false, "Should fail with no constraints")
    } catch {
        expect(true, "Correctly rejected no constraints")
    }
}

private func testFrErrorColumnOutOfRange() {
    do {
        _ = try GPUAIRConstraintCompiler.compile(
            numColumns: 2,
            logTraceLength: 2,
            transitions: [FrAIRExprBuilder.col(5) - FrAIRExprBuilder.col(0)],
            boundaries: [],
            traceGenerator: { [[Fr.zero], [Fr.zero]] })
        expect(false, "Should fail with column 5 out of range")
    } catch {
        expect(true, "Correctly rejected out-of-range column")
    }
}

private func testFrErrorUnknownPublicInput() {
    do {
        _ = try GPUAIRConstraintCompiler.compileExpression(
            FrAIRExprBuilder.publicInput("missing"), numColumns: 1)
        expect(false, "Should fail with unknown public input")
    } catch {
        expect(true, "Correctly rejected unknown public input")
    }
}
