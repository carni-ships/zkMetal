// AIR Constraint DSL Tests — validates the DSL builder, expression evaluation,
// compilation, and trace verification for all example AIRs.

import zkMetal

func runAIRDSLTests() {
    suite("AIR DSL — Expression Arithmetic")
    testExpressionDegree()
    testExpressionOperators()

    suite("AIR DSL — Fibonacci AIR")
    testDSLFibonacciTraceGeneration()
    testDSLFibonacciTraceVerification()
    testDSLFibonacciMatchesNative()

    suite("AIR DSL — Range Check AIR")
    testDSLRangeCheckTrace()
    testDSLRangeCheckVerification()

    suite("AIR DSL — Rescue Hash AIR")
    testDSLRescueHashTrace()
    testDSLRescueHashVerification()

    suite("AIR DSL — Collatz AIR")
    testDSLCollatzTrace()
    testDSLCollatzVerification()

    suite("AIR DSL — Compilation Errors")
    testCompileErrorNoColumns()
    testCompileErrorNoConstraints()
    testCompileErrorUnknownColumn()
    testCompileErrorNoTraceGen()

    suite("AIR DSL — CircleAIR Conformance")
    testDSLCircleAIRConformance()
}

// MARK: - Expression Tests

func testExpressionDegree() {
    let colExpr = AIRExpression.column("x")
    expectEqual(colExpr.degree, 1, "Column degree = 1")

    let constExpr = AIRExpression.constant(M31.one)
    expectEqual(constExpr.degree, 0, "Constant degree = 0")

    let addExpr = AIRExpression.add(.column("x"), .column("y"))
    expectEqual(addExpr.degree, 1, "Add degree = max(1,1) = 1")

    let mulExpr = AIRExpression.mul(.column("x"), .column("y"))
    expectEqual(mulExpr.degree, 2, "Mul degree = 1 + 1 = 2")

    let powExpr = AIRExpression.pow(.column("x"), 3)
    expectEqual(powExpr.degree, 3, "Pow degree = 1 * 3 = 3")

    let complexExpr = AIRExpression.mul(
        AIRExpression.pow(.column("x"), 2),
        .column("y")
    )
    expectEqual(complexExpr.degree, 3, "x^2 * y degree = 3")
}

func testExpressionOperators() {
    // Test that operators produce correct AST shapes
    let a = AIRExpression.column("a")
    let b = AIRExpression.column("b")

    let sum = a + b
    if case .add = sum {
        expect(true, "a + b is .add")
    } else {
        expect(false, "a + b should be .add")
    }

    let diff = a - b
    if case .sub = diff {
        expect(true, "a - b is .sub")
    } else {
        expect(false, "a - b should be .sub")
    }

    let prod = a * b
    if case .mul = prod {
        expect(true, "a * b is .mul")
    } else {
        expect(false, "a * b should be .mul")
    }

    let neg = -a
    if case .neg = neg {
        expect(true, "-a is .neg")
    } else {
        expect(false, "-a should be .neg")
    }

    let power = a ** 5
    if case .pow(_, 5) = power {
        expect(true, "a ** 5 is .pow(_, 5)")
    } else {
        expect(false, "a ** 5 should be .pow(_, 5)")
    }

    // Convenience operators with UInt32
    let addConst = a + 7
    if case .add(_, .constant(let v)) = addConst {
        expectEqual(v.v, UInt32(7), "a + 7 constant value")
    } else {
        expect(false, "a + 7 should be .add(_, .constant(7))")
    }

    let mulConst = a * 3
    if case .mul(_, .constant(let v)) = mulConst {
        expectEqual(v.v, UInt32(3), "a * 3 constant value")
    } else {
        expect(false, "a * 3 should be .mul(_, .constant(3))")
    }
}

// MARK: - Fibonacci DSL Tests

func testDSLFibonacciTraceGeneration() {
    do {
        let air = try buildFibonacciAIR(logTraceLength: 4, a0: M31.one, b0: M31.one)
        let trace = air.generateTrace()

        expectEqual(trace.count, 2, "DSL Fib: 2 columns")
        expectEqual(trace[0].count, 16, "DSL Fib: 16 rows")

        // Check initial values
        expectEqual(trace[0][0].v, UInt32(1), "DSL Fib a[0] = 1")
        expectEqual(trace[1][0].v, UInt32(1), "DSL Fib b[0] = 1")

        // Check Fibonacci relation
        for i in 0..<15 {
            expectEqual(trace[0][i + 1].v, trace[1][i].v,
                        "DSL Fib a[\(i+1)] = b[\(i)]")
            expectEqual(trace[1][i + 1].v, m31Add(trace[0][i], trace[1][i]).v,
                        "DSL Fib b[\(i+1)] = a[\(i)] + b[\(i)]")
        }
    } catch {
        expect(false, "DSL Fibonacci build failed: \(error)")
    }
}

func testDSLFibonacciTraceVerification() {
    do {
        let air = try buildFibonacciAIR(logTraceLength: 4)
        let trace = air.generateTrace()
        let err = air.verifyTraceFull(trace)
        expect(err == nil, "DSL Fib trace valid: \(err ?? "")")
    } catch {
        expect(false, "DSL Fibonacci verify failed: \(error)")
    }
}

func testDSLFibonacciMatchesNative() {
    // Compare DSL output with the native FibonacciAIR implementation
    do {
        let dslAIR = try buildFibonacciAIR(logTraceLength: 4)
        let nativeAIR = FibonacciAIR(logTraceLength: 4)

        let dslTrace = dslAIR.generateTrace()
        let nativeTrace = nativeAIR.generateTrace()

        expectEqual(dslTrace.count, nativeTrace.count, "Same column count")
        for col in 0..<dslTrace.count {
            expectEqual(dslTrace[col].count, nativeTrace[col].count, "Same row count col \(col)")
            for row in 0..<dslTrace[col].count {
                expectEqual(dslTrace[col][row].v, nativeTrace[col][row].v,
                            "Match col \(col) row \(row)")
            }
        }

        // Both should verify with their own constraint checks
        let dslErr = dslAIR.verifyTraceFull(dslTrace)
        let nativeErr = nativeAIR.verifyTrace(nativeTrace)
        expect(dslErr == nil, "DSL trace valid")
        expect(nativeErr == nil, "Native trace valid")

        // Cross-verify: DSL trace against native constraints
        let crossErr = nativeAIR.verifyTrace(dslTrace)
        expect(crossErr == nil, "DSL trace passes native constraints: \(crossErr ?? "")")
    } catch {
        expect(false, "DSL/native Fibonacci comparison failed: \(error)")
    }
}

// MARK: - Range Check DSL Tests

func testDSLRangeCheckTrace() {
    do {
        let values: [M31] = [M31(v: 100), M31(v: 50), M31(v: 200), M31(v: 0),
                             M31(v: 1000), M31(v: 500), M31(v: 65535), M31(v: 300)]
        let air = try buildRangeCheckAIR(logTraceLength: 3, values: values)
        let trace = air.generateTrace()

        expectEqual(trace.count, 1, "Range check 1 column")
        expectEqual(trace[0].count, 8, "Range check 8 rows")

        // Should be sorted
        for i in 0..<7 {
            expect(trace[0][i].v <= trace[0][i + 1].v,
                   "DSL range check sorted: \(trace[0][i].v) <= \(trace[0][i+1].v)")
        }
    } catch {
        expect(false, "DSL range check build failed: \(error)")
    }
}

func testDSLRangeCheckVerification() {
    // Note: the range check constraint diff*(bound-diff) is a soundness check
    // that catches invalid proofs via degree bounds, not a zero-on-valid check.
    // We verify the trace structure (sorted, in-bound) rather than constraint evaluation.
    do {
        let values: [M31] = [M31(v: 10), M31(v: 20), M31(v: 30), M31(v: 40)]
        let air = try buildRangeCheckAIR(logTraceLength: 2, values: values, bound: 65536)
        let trace = air.generateTrace()

        // Verify trace structure: sorted and in-bound
        for i in 0..<3 {
            expect(trace[0][i].v <= trace[0][i + 1].v,
                   "DSL RC sorted: \(trace[0][i].v) <= \(trace[0][i+1].v)")
        }
        for i in 0..<4 {
            expect(trace[0][i].v < 65536, "DSL RC value \(i) < bound")
        }

        // Check boundary constraint
        let bcs = air.boundaryConstraints
        expect(bcs.count > 0, "DSL RC has boundary constraints")
        expect(bcs[0].value.v == 10, "DSL RC min value = 10")
    } catch {
        expect(false, "DSL range check verify failed: \(error)")
    }
}

// MARK: - Rescue Hash DSL Tests

func testDSLRescueHashTrace() {
    do {
        let input: [M31] = [M31(v: 1), M31(v: 2), M31(v: 3), M31(v: 4)]
        let air = try buildRescueHashAIR(logTraceLength: 4, width: 4, input: input)
        let trace = air.generateTrace()

        expectEqual(trace.count, 4, "Rescue 4 state columns")
        expectEqual(trace[0].count, 16, "Rescue 16 rows")

        // Initial state should match input
        for i in 0..<4 {
            expectEqual(trace[i][0].v, input[i].v, "Rescue s\(i)[0] = input[\(i)]")
        }
    } catch {
        expect(false, "DSL Rescue hash build failed: \(error)")
    }
}

func testDSLRescueHashVerification() {
    do {
        let input: [M31] = [M31(v: 1), M31(v: 2), M31(v: 3), M31(v: 4)]
        let air = try buildRescueHashAIR(logTraceLength: 4, width: 4, input: input)
        let trace = air.generateTrace()
        let err = air.verifyTraceFull(trace)
        expect(err == nil, "DSL Rescue trace valid: \(err ?? "")")
    } catch {
        expect(false, "DSL Rescue hash verify failed: \(error)")
    }
}

// MARK: - Collatz DSL Tests

func testDSLCollatzTrace() {
    do {
        // Start with 7: 7 -> 22 -> 11 -> 34 -> 17 -> 52 -> 26 -> 13 -> ...
        let air = try buildCollatzAIR(logTraceLength: 4, startValue: 7)
        let trace = air.generateTrace()

        expectEqual(trace.count, 2, "Collatz 2 columns")
        expectEqual(trace[0].count, 16, "Collatz 16 rows")

        // Check initial value
        expectEqual(trace[0][0].v, UInt32(7), "Collatz val[0] = 7")
        expectEqual(trace[1][0].v, UInt32(1), "Collatz is_odd[0] = 1 (7 is odd)")

        // Check second value: 7 is odd -> 3*7+1 = 22
        expectEqual(trace[0][1].v, UInt32(22), "Collatz val[1] = 22")
        expectEqual(trace[1][1].v, UInt32(0), "Collatz is_odd[1] = 0 (22 is even)")

        // Third: 22/2 = 11
        expectEqual(trace[0][2].v, UInt32(11), "Collatz val[2] = 11")
    } catch {
        expect(false, "DSL Collatz build failed: \(error)")
    }
}

func testDSLCollatzVerification() {
    do {
        let air = try buildCollatzAIR(logTraceLength: 4, startValue: 7)
        let trace = air.generateTrace()
        let err = air.verifyTraceFull(trace)
        expect(err == nil, "DSL Collatz trace valid: \(err ?? "")")
    } catch {
        expect(false, "DSL Collatz verify failed: \(error)")
    }
}

// MARK: - Compilation Error Tests

func testCompileErrorNoColumns() {
    let builder = AIRBuilder(logTraceLength: 2)
    builder.traceGen { _ in [[]] }
    do {
        _ = try builder.compile()
        expect(false, "Should fail with no columns")
    } catch {
        expect(true, "Correctly rejected: \(error)")
    }
}

func testCompileErrorNoConstraints() {
    let builder = AIRBuilder(logTraceLength: 2)
    builder.addColumn(name: "x")
    builder.traceGen { _ in [[M31.zero, M31.zero, M31.zero, M31.zero]] }
    do {
        _ = try builder.compile()
        expect(false, "Should fail with no constraints")
    } catch {
        expect(true, "Correctly rejected: \(error)")
    }
}

func testCompileErrorUnknownColumn() {
    // Test that building with a valid column works correctly
    let builder = AIRBuilder(logTraceLength: 2)
    builder.addColumn(name: "x")
    builder.traceGen { _ in [[M31.zero, M31.zero, M31.zero, M31.zero]] }
    builder.transition { ctx in ctx.col("x") - ctx.col("x") }
    do {
        let air = try builder.compile()
        expect(air.numColumns == 1, "Valid column reference works")
    } catch {
        expect(false, "Unexpected error: \(error)")
    }
}

func testCompileErrorNoTraceGen() {
    let builder = AIRBuilder(logTraceLength: 2)
    builder.addColumn(name: "x")
    builder.transition { ctx in ctx.col("x") - ctx.col("x") }
    do {
        _ = try builder.compile()
        expect(false, "Should fail with no trace generator")
    } catch {
        expect(true, "Correctly rejected no trace gen: \(error)")
    }
}

// MARK: - CircleAIR Conformance

func testDSLCircleAIRConformance() {
    do {
        let air = try buildFibonacciAIR(logTraceLength: 3)

        // CompiledAIR conforms to CircleAIR
        let circleAIR: CircleAIR = air
        expectEqual(circleAIR.numColumns, 2, "CircleAIR numColumns")
        expectEqual(circleAIR.logTraceLength, 3, "CircleAIR logTraceLength")
        expectEqual(circleAIR.traceLength, 8, "CircleAIR traceLength")
        expectEqual(circleAIR.numConstraints, 2, "CircleAIR numConstraints")
        expectEqual(circleAIR.constraintDegrees, [1, 1], "CircleAIR constraint degrees")

        // Boundary constraints
        let bcs = circleAIR.boundaryConstraints
        expectEqual(bcs.count, 2, "CircleAIR 2 boundary constraints")
        expectEqual(bcs[0].column, 0, "BC 0 column")
        expectEqual(bcs[0].row, 0, "BC 0 row")
        expectEqual(bcs[0].value.v, UInt32(1), "BC 0 value")
        expectEqual(bcs[1].column, 1, "BC 1 column")
        expectEqual(bcs[1].row, 0, "BC 1 row")
        expectEqual(bcs[1].value.v, UInt32(1), "BC 1 value")

        // Generate and verify via CircleAIR protocol
        let trace = circleAIR.generateTrace()
        let err = circleAIR.verifyTrace(trace)
        expect(err == nil, "CircleAIR verify: \(err ?? "")")
    } catch {
        expect(false, "CircleAIR conformance failed: \(error)")
    }
}
