// AIR Constraint Compiler Tests — validates the compiler, standard library,
// degree analysis, composition, and CircleAIR conformance.

import zkMetal

public func runAIRConstraintCompilerTests() {
    suite("AIR Compiler — Index-Based Expression Builder")
    testExprBuilderCol()
    testExprBuilderNext()
    testExprBuilderConstant()
    testExprBuilderArithmetic()

    suite("AIR Compiler — Expression Compilation")
    testCompileSimpleExpression()
    testCompileComplexExpression()
    testCompileWithPublicInputs()

    suite("AIR Compiler — Degree Analysis")
    testDegreeAnalysisLinear()
    testDegreeAnalysisQuadratic()
    testDegreeAnalysisCubic()
    testDegreeAnalysisMixed()

    suite("AIR Compiler — Composition")
    testCompositionSingleConstraint()
    testCompositionMultipleConstraints()

    suite("AIR Compiler — Constant Folding")
    testConstantFoldingAddition()
    testConstantFoldingMultiplication()
    testConstantFoldingIdentities()
    testConstantFoldingPower()

    suite("AIR Compiler — Fibonacci Standard Library")
    testStdFibonacciTrace()
    testStdFibonacciVerification()
    testStdFibonacciMatchesNative()

    suite("AIR Compiler — Range Check Standard Library")
    testStdRangeCheckTrace()
    testStdRangeCheckSorted()
    testStdRangeCheckDegree()

    suite("AIR Compiler — Permutation Check Standard Library")
    testStdPermutationValidTrace()
    testStdPermutationBoundary()

    suite("AIR Compiler — Memory Consistency Standard Library")
    testStdMemoryConsistencyTrace()
    testStdMemoryConsistencyReadAfterWrite()

    suite("AIR Compiler — CircleAIR Conformance")
    testCompiledAIRProtocolConformance()
    testCompiledAIRTraceVerify()

    suite("AIR Compiler — Error Handling")
    testCompilerErrorNoColumns()
    testCompilerErrorNoConstraints()
    testCompilerErrorUnknownColumn()
}

// MARK: - Index-Based Expression Builder

func testExprBuilderCol() {
    let expr = AIRExprBuilder.col(0)
    if case .column("__col_0") = expr {
        expect(true, "col(0) produces .column(\"__col_0\")")
    } else {
        expect(false, "col(0) should produce .column(\"__col_0\")")
    }

    let expr2 = AIRExprBuilder.col(5)
    if case .column("__col_5") = expr2 {
        expect(true, "col(5) produces .column(\"__col_5\")")
    } else {
        expect(false, "col(5) should produce .column(\"__col_5\")")
    }
}

func testExprBuilderNext() {
    let expr = AIRExprBuilder.next(1)
    if case .nextColumn("__col_1") = expr {
        expect(true, "next(1) produces .nextColumn(\"__col_1\")")
    } else {
        expect(false, "next(1) should produce .nextColumn(\"__col_1\")")
    }
}

func testExprBuilderConstant() {
    let expr = AIRExprBuilder.constant(42)
    if case .constant(let v) = expr {
        expectEqual(v.v, UInt32(42), "constant(42) value")
    } else {
        expect(false, "constant(42) should produce .constant")
    }

    let expr2 = AIRExprBuilder.constant(M31(v: 100))
    if case .constant(let v) = expr2 {
        expectEqual(v.v, UInt32(100), "constant(M31(100)) value")
    } else {
        expect(false, "constant(M31(100)) should produce .constant")
    }
}

func testExprBuilderArithmetic() {
    let a = AIRExprBuilder.col(0)
    let b = AIRExprBuilder.col(1)

    // Test addition
    let sum = a + b
    expectEqual(sum.degree, 1, "col(0) + col(1) degree = 1")

    // Test subtraction
    let diff = AIRExprBuilder.next(0) - b
    expectEqual(diff.degree, 1, "next(0) - col(1) degree = 1")
    expect(diff.usesNextRow, "next(0) - col(1) uses next row")

    // Test multiplication
    let prod = a * b
    expectEqual(prod.degree, 2, "col(0) * col(1) degree = 2")

    // Test power
    let cubed = a ** 3
    expectEqual(cubed.degree, 3, "col(0)^3 degree = 3")
}

// MARK: - Expression Compilation

func testCompileSimpleExpression() {
    // Compile: col(0) - col(1)
    let expr = AIRExprBuilder.col(0) - AIRExprBuilder.col(1)
    do {
        let compiled = try AIRConstraintCompiler.compileExpression(
            expr, columnIndex: ["__col_0": 0, "__col_1": 1])
        let result = compiled.evaluate([M31(v: 5), M31(v: 3)], [M31.zero, M31.zero])
        expectEqual(result.v, UInt32(2), "5 - 3 = 2")

        let result2 = compiled.evaluate([M31(v: 10), M31(v: 10)], [M31.zero, M31.zero])
        expectEqual(result2.v, UInt32(0), "10 - 10 = 0")
    } catch {
        expect(false, "Compile failed: \(error)")
    }
}

func testCompileComplexExpression() {
    // Compile: next(0) - (col(0) + col(1))  [Fibonacci b' = a + b]
    let expr = AIRExprBuilder.next(0) - (AIRExprBuilder.col(0) + AIRExprBuilder.col(1))
    do {
        let compiled = try AIRConstraintCompiler.compileExpression(
            expr, columnIndex: ["__col_0": 0, "__col_1": 1])

        // Fibonacci: current = [1, 1], next = [1, 2] => next(0) - (1+1) = 1 - 2 = -1
        let result = compiled.evaluate([M31(v: 1), M31(v: 1)], [M31(v: 1), M31(v: 2)])
        // next[0] - (current[0] + current[1]) = 1 - (1+1) = 1 - 2 = -1 mod p
        expectEqual(result.v, M31.P - 1, "1 - 2 = -1 mod p")

        // Valid case: current = [1, 1], next = [1, 2] => next[0]=1, 1+1=2, diff=-1
        // Actually valid Fib: current=[1,1], next=[1,2], check next[1]-(1+1) = 2-2 = 0
        let result2 = compiled.evaluate([M31(v: 1), M31(v: 1)], [M31(v: 2), M31(v: 0)])
        expectEqual(result2.v, UInt32(0), "next(0)=2, col(0)+col(1)=2, diff=0")
    } catch {
        expect(false, "Compile failed: \(error)")
    }
}

func testCompileWithPublicInputs() {
    // Compile: col(0) - pub("init")
    let expr = AIRExpression.sub(.column("__col_0"), .publicInput("init"))
    do {
        let compiled = try AIRConstraintCompiler.compileExpression(
            expr,
            columnIndex: ["__col_0": 0],
            publicInputValues: ["init": M31(v: 42)])
        let result = compiled.evaluate([M31(v: 42)], [M31.zero])
        expectEqual(result.v, UInt32(0), "col(0) = 42 = pub(init), constraint satisfied")

        let result2 = compiled.evaluate([M31(v: 10)], [M31.zero])
        expect(result2.v != 0, "col(0) = 10 != 42 = pub(init), constraint violated")
    } catch {
        expect(false, "Compile with pub inputs failed: \(error)")
    }
}

// MARK: - Degree Analysis

func testDegreeAnalysisLinear() {
    let analysis = AIRConstraintCompiler.analyzeDegrees(
        [AIRExprBuilder.next(0) - AIRExprBuilder.col(1)],
        logTraceLength: 4)
    expectEqual(analysis.transitionDegrees, [1], "Linear constraint degree = [1]")
    expectEqual(analysis.maxTransitionDegree, 1, "Max degree = 1")
    expectEqual(analysis.compositionDegreeBound, 16, "Composition bound = 1 * 16")
    expectEqual(analysis.numQuotientChunks, 1, "1 quotient chunk for linear")
}

func testDegreeAnalysisQuadratic() {
    // diff * (bound - diff) has degree 2
    let diff = AIRExprBuilder.next(0) - AIRExprBuilder.col(0)
    let constraint = diff * (AIRExprBuilder.constant(65536) - diff)
    let analysis = AIRConstraintCompiler.analyzeDegrees([constraint], logTraceLength: 3)
    expectEqual(analysis.maxTransitionDegree, 2, "Quadratic max degree = 2")
    expectEqual(analysis.numQuotientChunks, 2, "2 quotient chunks for quadratic")
}

func testDegreeAnalysisCubic() {
    // x^3 has degree 3
    let constraint = AIRExprBuilder.col(0) ** 3
    let analysis = AIRConstraintCompiler.analyzeDegrees([constraint], logTraceLength: 4)
    expectEqual(analysis.maxTransitionDegree, 3, "Cubic max degree = 3")
    expectEqual(analysis.compositionDegreeBound, 48, "Composition bound = 3 * 16")
    expectEqual(analysis.numQuotientChunks, 3, "3 quotient chunks for cubic")
}

func testDegreeAnalysisMixed() {
    let constraints: [AIRExpression] = [
        AIRExprBuilder.next(0) - AIRExprBuilder.col(1),         // degree 1
        AIRExprBuilder.col(0) * AIRExprBuilder.col(1),          // degree 2
        AIRExprBuilder.col(0) ** 3                              // degree 3
    ]
    let analysis = AIRConstraintCompiler.analyzeDegrees(constraints, logTraceLength: 4)
    expectEqual(analysis.transitionDegrees, [1, 2, 3], "Mixed degrees")
    expectEqual(analysis.maxTransitionDegree, 3, "Max of mixed = 3")
}

// MARK: - Composition

func testCompositionSingleConstraint() {
    do {
        let expr = AIRExprBuilder.col(0) - AIRExprBuilder.constant(5)
        let compiled = try AIRConstraintCompiler.compileExpression(
            expr, columnIndex: ["__col_0": 0])
        let composed = AIRConstraintCompiler.composeConstraints(
            [compiled], alpha: M31(v: 7))

        // col(0) = 5 => constraint = 0
        let result = composed([M31(v: 5)], [M31.zero])
        expectEqual(result.v, UInt32(0), "Single constraint, satisfied => 0")

        // col(0) = 3 => constraint = 3 - 5 = -2 mod p
        let result2 = composed([M31(v: 3)], [M31.zero])
        expectEqual(result2.v, M31.P - 2, "Single constraint, violated => -2")
    } catch {
        expect(false, "Composition failed: \(error)")
    }
}

func testCompositionMultipleConstraints() {
    do {
        // C0: col(0) - 1 = 0
        // C1: col(1) - 2 = 0
        let c0 = try AIRConstraintCompiler.compileExpression(
            AIRExprBuilder.col(0) - AIRExprBuilder.constant(1),
            columnIndex: ["__col_0": 0, "__col_1": 1])
        let c1 = try AIRConstraintCompiler.compileExpression(
            AIRExprBuilder.col(1) - AIRExprBuilder.constant(2),
            columnIndex: ["__col_0": 0, "__col_1": 1])

        let alpha = M31(v: 3)
        let composed = AIRConstraintCompiler.composeConstraints([c0, c1], alpha: alpha)

        // When both satisfied: col(0)=1, col(1)=2
        let result = composed([M31(v: 1), M31(v: 2)], [M31.zero, M31.zero])
        expectEqual(result.v, UInt32(0), "Both constraints satisfied => 0")

        // When C0 violated: col(0)=5 => C0=4, C1=0 => 1*4 + 3*0 = 4
        let result2 = composed([M31(v: 5), M31(v: 2)], [M31.zero, M31.zero])
        expectEqual(result2.v, UInt32(4), "C0=4, C1=0, alpha=3: 1*4 + 3*0 = 4")

        // When C1 violated: col(0)=1, col(1)=5 => C0=0, C1=3 => 1*0 + 3*3 = 9
        let result3 = composed([M31(v: 1), M31(v: 5)], [M31.zero, M31.zero])
        expectEqual(result3.v, UInt32(9), "C0=0, C1=3, alpha=3: 1*0 + 3*3 = 9")
    } catch {
        expect(false, "Multi-composition failed: \(error)")
    }
}

// MARK: - Constant Folding

func testConstantFoldingAddition() {
    let expr = AIRExpression.add(.constant(M31(v: 3)), .constant(M31(v: 4)))
    let optimized = AIRConstraintCompiler.optimizeExpression(expr)
    if case .constant(let v) = optimized {
        expectEqual(v.v, UInt32(7), "3 + 4 folded to 7")
    } else {
        expect(false, "Should fold to constant")
    }
}

func testConstantFoldingMultiplication() {
    let expr = AIRExpression.mul(.constant(M31(v: 5)), .constant(M31(v: 6)))
    let optimized = AIRConstraintCompiler.optimizeExpression(expr)
    if case .constant(let v) = optimized {
        expectEqual(v.v, UInt32(30), "5 * 6 folded to 30")
    } else {
        expect(false, "Should fold to constant")
    }
}

func testConstantFoldingIdentities() {
    let col = AIRExpression.column("x")

    // 0 + x = x
    let addZero = AIRConstraintCompiler.optimizeExpression(
        .add(.constant(M31.zero), col))
    if case .column("x") = addZero {
        expect(true, "0 + x = x")
    } else {
        expect(false, "0 + x should fold to x")
    }

    // x * 1 = x
    let mulOne = AIRConstraintCompiler.optimizeExpression(
        .mul(col, .constant(M31.one)))
    if case .column("x") = mulOne {
        expect(true, "x * 1 = x")
    } else {
        expect(false, "x * 1 should fold to x")
    }

    // x * 0 = 0
    let mulZero = AIRConstraintCompiler.optimizeExpression(
        .mul(col, .constant(M31.zero)))
    if case .constant(let v) = mulZero {
        expectEqual(v.v, UInt32(0), "x * 0 = 0")
    } else {
        expect(false, "x * 0 should fold to 0")
    }

    // x - 0 = x
    let subZero = AIRConstraintCompiler.optimizeExpression(
        .sub(col, .constant(M31.zero)))
    if case .column("x") = subZero {
        expect(true, "x - 0 = x")
    } else {
        expect(false, "x - 0 should fold to x")
    }
}

func testConstantFoldingPower() {
    // constant^n = constant
    let expr = AIRExpression.pow(.constant(M31(v: 2)), 10)
    let optimized = AIRConstraintCompiler.optimizeExpression(expr)
    if case .constant(let v) = optimized {
        expectEqual(v.v, UInt32(1024), "2^10 folded to 1024")
    } else {
        expect(false, "constant^n should fold")
    }

    // x^0 = 1
    let powZero = AIRConstraintCompiler.optimizeExpression(
        .pow(.column("x"), 0))
    if case .constant(let v) = powZero {
        expectEqual(v.v, UInt32(1), "x^0 = 1")
    } else {
        expect(false, "x^0 should fold to 1")
    }

    // x^1 = x
    let powOne = AIRConstraintCompiler.optimizeExpression(
        .pow(.column("x"), 1))
    if case .column("x") = powOne {
        expect(true, "x^1 = x")
    } else {
        expect(false, "x^1 should fold to x")
    }
}

// MARK: - Fibonacci Standard Library

func testStdFibonacciTrace() {
    do {
        let air = try AIRStandardLibrary.fibonacci(logTraceLength: 4)
        let trace = air.generateTrace()

        expectEqual(trace.count, 2, "Std Fib: 2 columns")
        expectEqual(trace[0].count, 16, "Std Fib: 16 rows")

        // Check initial values
        expectEqual(trace[0][0].v, UInt32(1), "Std Fib a[0] = 1")
        expectEqual(trace[1][0].v, UInt32(1), "Std Fib b[0] = 1")

        // Check Fibonacci relation
        for i in 0..<15 {
            expectEqual(trace[0][i + 1].v, trace[1][i].v,
                        "Std Fib a[\(i+1)] = b[\(i)]")
            expectEqual(trace[1][i + 1].v, m31Add(trace[0][i], trace[1][i]).v,
                        "Std Fib b[\(i+1)] = a[\(i)] + b[\(i)]")
        }
    } catch {
        expect(false, "Std Fibonacci build failed: \(error)")
    }
}

func testStdFibonacciVerification() {
    do {
        let air = try AIRStandardLibrary.fibonacci(logTraceLength: 4)
        let trace = air.generateTrace()
        let err = air.verifyTraceFull(trace)
        expect(err == nil, "Std Fib trace valid: \(err ?? "")")
    } catch {
        expect(false, "Std Fibonacci verify failed: \(error)")
    }
}

func testStdFibonacciMatchesNative() {
    do {
        let stdAIR = try AIRStandardLibrary.fibonacci(logTraceLength: 4)
        let nativeAIR = FibonacciAIR(logTraceLength: 4)

        let stdTrace = stdAIR.generateTrace()
        let nativeTrace = nativeAIR.generateTrace()

        // Compare traces element by element
        expectEqual(stdTrace.count, nativeTrace.count, "Same column count")
        for col in 0..<stdTrace.count {
            for row in 0..<stdTrace[col].count {
                expectEqual(stdTrace[col][row].v, nativeTrace[col][row].v,
                            "Match col \(col) row \(row)")
            }
        }

        // Cross-verify: std trace against native constraints
        let crossErr = nativeAIR.verifyTrace(stdTrace)
        expect(crossErr == nil, "Std trace passes native constraints: \(crossErr ?? "")")
    } catch {
        expect(false, "Std/native Fibonacci comparison failed: \(error)")
    }
}

// MARK: - Range Check Standard Library

func testStdRangeCheckTrace() {
    do {
        let values: [M31] = [M31(v: 100), M31(v: 50), M31(v: 200), M31(v: 0)]
        let air = try AIRStandardLibrary.rangeCheck(logTraceLength: 2, values: values)
        let trace = air.generateTrace()

        expectEqual(trace.count, 1, "Range check 1 column")
        expectEqual(trace[0].count, 4, "Range check 4 rows")

        // First value should be minimum
        expectEqual(trace[0][0].v, UInt32(0), "RC min value = 0")
    } catch {
        expect(false, "Std range check build failed: \(error)")
    }
}

func testStdRangeCheckSorted() {
    do {
        let values: [M31] = [M31(v: 500), M31(v: 100), M31(v: 300), M31(v: 200),
                              M31(v: 800), M31(v: 400), M31(v: 700), M31(v: 600)]
        let air = try AIRStandardLibrary.rangeCheck(logTraceLength: 3, values: values)
        let trace = air.generateTrace()

        // Should be sorted
        for i in 0..<7 {
            expect(trace[0][i].v <= trace[0][i + 1].v,
                   "RC sorted: \(trace[0][i].v) <= \(trace[0][i+1].v)")
        }
    } catch {
        expect(false, "Std range check sorted failed: \(error)")
    }
}

func testStdRangeCheckDegree() {
    do {
        let values: [M31] = [M31(v: 10), M31(v: 20), M31(v: 30), M31(v: 40)]
        let air = try AIRStandardLibrary.rangeCheck(logTraceLength: 2, values: values)

        // Range check constraint is quadratic: diff * (bound - diff)
        expectEqual(air.constraintDegrees.count, 1, "RC 1 constraint")
        expectEqual(air.constraintDegrees[0], 2, "RC constraint degree = 2")
    } catch {
        expect(false, "Std range check degree failed: \(error)")
    }
}

// MARK: - Permutation Check Standard Library

func testStdPermutationValidTrace() {
    do {
        let n = 4
        let valuesA: [M31] = [M31(v: 1), M31(v: 2), M31(v: 3), M31(v: 4)]
        let valuesB: [M31] = [M31(v: 3), M31(v: 1), M31(v: 4), M31(v: 2)] // permutation of A
        let gamma = M31(v: 17)

        let air = try AIRStandardLibrary.permutationCheck(
            logTraceLength: 2, valuesA: valuesA, valuesB: valuesB, gamma: gamma)
        let trace = air.generateTrace()

        expectEqual(trace.count, 3, "Perm check: 3 columns (A, B, acc)")
        expectEqual(trace[0].count, n, "Perm check: \(n) rows")

        // Accumulator starts at 1
        expectEqual(trace[2][0].v, UInt32(1), "Perm acc[0] = 1")

        // Values A should match input
        for i in 0..<n {
            expectEqual(trace[0][i].v, valuesA[i].v, "A[\(i)] matches")
        }
    } catch {
        expect(false, "Std permutation build failed: \(error)")
    }
}

func testStdPermutationBoundary() {
    do {
        let valuesA: [M31] = [M31(v: 10), M31(v: 20), M31(v: 30), M31(v: 40)]
        let valuesB: [M31] = [M31(v: 20), M31(v: 40), M31(v: 10), M31(v: 30)]
        let gamma = M31(v: 5)

        let air = try AIRStandardLibrary.permutationCheck(
            logTraceLength: 2, valuesA: valuesA, valuesB: valuesB, gamma: gamma)

        let bcs = air.boundaryConstraints
        expect(bcs.count >= 1, "Perm has boundary constraint for acc[0]")
        // Find the boundary for column 2 (accumulator), row 0
        let accBC = bcs.first { $0.column == 2 && $0.row == 0 }
        expect(accBC != nil, "Boundary: acc[0] exists")
        if let bc = accBC {
            expectEqual(bc.value.v, UInt32(1), "Boundary: acc[0] = 1")
        }
    } catch {
        expect(false, "Std permutation boundary failed: \(error)")
    }
}

// MARK: - Memory Consistency Standard Library

func testStdMemoryConsistencyTrace() {
    do {
        let ops: [(address: UInt32, value: M31, isWrite: Bool, timestamp: UInt32)] = [
            (address: 0, value: M31(v: 42), isWrite: true, timestamp: 0),
            (address: 0, value: M31(v: 42), isWrite: false, timestamp: 1),
            (address: 1, value: M31(v: 99), isWrite: true, timestamp: 2),
            (address: 1, value: M31(v: 99), isWrite: false, timestamp: 3)
        ]

        let air = try AIRStandardLibrary.memoryConsistency(
            logTraceLength: 2, operations: ops)
        let trace = air.generateTrace()

        expectEqual(trace.count, 4, "Memory: 4 columns (addr, val, is_write, ts)")
        expectEqual(trace[0].count, 4, "Memory: 4 rows")

        // Should be sorted by address
        for i in 0..<3 {
            expect(trace[0][i].v <= trace[0][i + 1].v,
                   "Memory sorted by address: \(trace[0][i].v) <= \(trace[0][i+1].v)")
        }
    } catch {
        expect(false, "Std memory consistency build failed: \(error)")
    }
}

func testStdMemoryConsistencyReadAfterWrite() {
    do {
        let ops: [(address: UInt32, value: M31, isWrite: Bool, timestamp: UInt32)] = [
            (address: 5, value: M31(v: 100), isWrite: true, timestamp: 0),
            (address: 5, value: M31(v: 100), isWrite: false, timestamp: 1),
            (address: 5, value: M31(v: 100), isWrite: false, timestamp: 2),
            (address: 5, value: M31(v: 100), isWrite: false, timestamp: 3)
        ]

        let air = try AIRStandardLibrary.memoryConsistency(
            logTraceLength: 2, operations: ops)
        let trace = air.generateTrace()

        // All reads after write should have same value
        for i in 0..<4 {
            expectEqual(trace[1][i].v, UInt32(100), "Memory val[\(i)] = 100")
        }

        // All same address
        for i in 0..<4 {
            expectEqual(trace[0][i].v, UInt32(5), "Memory addr[\(i)] = 5")
        }

        // Verify the trace satisfies the read-after-write constraint
        let err = air.verifyTraceFull(trace)
        expect(err == nil, "Memory RAW trace valid: \(err ?? "")")
    } catch {
        expect(false, "Std memory RAW failed: \(error)")
    }
}

// MARK: - CircleAIR Conformance

func testCompiledAIRProtocolConformance() {
    do {
        let air = try AIRStandardLibrary.fibonacci(logTraceLength: 3)

        // CompiledAIR conforms to CircleAIR
        let circleAIR: CircleAIR = air
        expectEqual(circleAIR.numColumns, 2, "CircleAIR numColumns = 2")
        expectEqual(circleAIR.logTraceLength, 3, "CircleAIR logTraceLength = 3")
        expectEqual(circleAIR.traceLength, 8, "CircleAIR traceLength = 8")
        expectEqual(circleAIR.numConstraints, 2, "CircleAIR numConstraints = 2")

        // Constraint degrees
        let degrees = circleAIR.constraintDegrees
        expectEqual(degrees.count, 2, "2 constraint degrees")
        expectEqual(degrees[0], 1, "Fib constraint 0 degree = 1")
        expectEqual(degrees[1], 1, "Fib constraint 1 degree = 1")

        // Boundary constraints
        let bcs = circleAIR.boundaryConstraints
        expectEqual(bcs.count, 2, "2 boundary constraints")
        expectEqual(bcs[0].column, 0, "BC 0 column = 0")
        expectEqual(bcs[0].row, 0, "BC 0 row = 0")
        expectEqual(bcs[0].value.v, UInt32(1), "BC 0 value = 1")
        expectEqual(bcs[1].column, 1, "BC 1 column = 1")
        expectEqual(bcs[1].value.v, UInt32(1), "BC 1 value = 1")
    } catch {
        expect(false, "CircleAIR conformance failed: \(error)")
    }
}

func testCompiledAIRTraceVerify() {
    do {
        let air = try AIRStandardLibrary.fibonacci(logTraceLength: 4)
        let trace = air.generateTrace()

        // Verify via CircleAIR protocol (verifyTrace uses evaluateConstraints)
        let circleAIR: CircleAIR = air
        let err = circleAIR.verifyTrace(trace)
        expect(err == nil, "CircleAIR.verifyTrace passes: \(err ?? "")")

        // Also verify via CompiledAIR's verifyTraceFull
        let fullErr = air.verifyTraceFull(trace)
        expect(fullErr == nil, "CompiledAIR.verifyTraceFull passes: \(fullErr ?? "")")
    } catch {
        expect(false, "CircleAIR trace verify failed: \(error)")
    }
}

// MARK: - Error Handling

func testCompilerErrorNoColumns() {
    do {
        _ = try AIRConstraintCompiler.compile(
            numColumns: 0,
            logTraceLength: 2,
            transitions: [AIRExprBuilder.col(0)],
            boundaries: [],
            traceGenerator: { [[]] })
        expect(false, "Should fail with 0 columns")
    } catch {
        expect(true, "Correctly rejected 0 columns: \(error)")
    }
}

func testCompilerErrorNoConstraints() {
    do {
        _ = try AIRConstraintCompiler.compile(
            numColumns: 1,
            logTraceLength: 2,
            transitions: [],
            boundaries: [],
            traceGenerator: { [[M31.zero]] })
        expect(false, "Should fail with no constraints")
    } catch {
        expect(true, "Correctly rejected no constraints: \(error)")
    }
}

func testCompilerErrorUnknownColumn() {
    // Using col(5) in a 2-column AIR should fail
    do {
        _ = try AIRConstraintCompiler.compile(
            numColumns: 2,
            logTraceLength: 2,
            transitions: [AIRExprBuilder.col(5) - AIRExprBuilder.col(0)],
            boundaries: [],
            traceGenerator: { [[M31.zero], [M31.zero]] })
        expect(false, "Should fail with unknown column __col_5")
    } catch {
        expect(true, "Correctly rejected unknown column: \(error)")
    }
}
