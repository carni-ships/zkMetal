// GPU AIR Trace Validator Tests — validates transition checking, boundary verification,
// periodic column evaluation, degree bound checking, composition polynomial validation,
// and multi-register trace validation.

import Foundation
import zkMetal

public func runGPUAIRTraceValidatorTests() {
    suite("GPUAIRTraceValidator - Construction")
    testValidatorFromAIR()
    testValidatorFromExpressions()
    testValidatorDimensions()

    suite("GPUAIRTraceValidator - Transition Constraints")
    testFibonacciTransitionValid()
    testFibonacciTransitionInvalid()
    testCounterTransition()
    testQuadraticTransition()

    suite("GPUAIRTraceValidator - Boundary Constraints")
    testBoundaryValid()
    testBoundaryInvalid()
    testMultipleBoundaries()
    testBoundaryOutOfRange()

    suite("GPUAIRTraceValidator - Periodic Constraints")
    testPeriodicEveryOtherRow()
    testPeriodicWithPhase()
    testPeriodicNoViolations()

    suite("GPUAIRTraceValidator - Degree Bound Checking")
    testDegreeBoundLinearTrace()
    testDegreeBoundQuadraticTrace()
    testDegreeBoundExceeded()

    suite("GPUAIRTraceValidator - Composition Polynomial")
    testCompositionValid()
    testCompositionInvalid()
    testValidatorCompositionMultipleConstraints()

    suite("GPUAIRTraceValidator - Multi-Register Validation")
    testMultiRegisterValid()
    testMultiRegisterCrossConstraint()
    testMultiRegisterMixed()

    suite("GPUAIRTraceValidator - Full Validation Report")
    testFullReportValid()
    testFullReportInvalid()
    testFullReportSummary()

    suite("GPUAIRTraceValidator - Batch Evaluation")
    testBatchEvalAllZero()
    testBatchEvalWithViolations()

    suite("GPUAIRTraceValidator - Degree Analysis")
    testValidatorDegreeAnalysisLinear()
    testValidatorDegreeAnalysisMixed()

    suite("GPUAIRTraceValidator - Edge Cases")
    testSingleRowConstraint()
    testCountViolations()
    testEvaluateAtSpecificRow()
}

// MARK: - Helper: Generate Fibonacci Trace

private func generateFibTrace(logN: Int, a0: Fr = Fr.one, b0: Fr = Fr.one) -> [[Fr]] {
    let n = 1 << logN
    var colA = [Fr](repeating: Fr.zero, count: n)
    var colB = [Fr](repeating: Fr.zero, count: n)
    colA[0] = a0
    colB[0] = b0
    for i in 1..<n {
        colA[i] = colB[i - 1]
        colB[i] = frAdd(colA[i - 1], colB[i - 1])
    }
    return [colA, colB]
}

// MARK: - Helper: Generate Counter Trace

private func generateCounterTrace(logN: Int) -> [[Fr]] {
    let n = 1 << logN
    var col = [Fr](repeating: Fr.zero, count: n)
    for i in 0..<n {
        col[i] = frFromInt(UInt64(i))
    }
    return [col]
}

// MARK: - Construction

private func testValidatorFromAIR() {
    do {
        let air = try FrAIRStandardLibrary.fibonacci(logTraceLength: 4)
        let validator = GPUAIRTraceValidatorEngine.fromAIR(air)
        expectEqual(validator.numColumns, 2, "Fibonacci: 2 columns")
        expectEqual(validator.logTraceLength, 4, "logN = 4")
        expectEqual(validator.traceLength, 16, "N = 16")
        expectEqual(validator.transitionConstraints.count, 2, "2 transition constraints")
        expectEqual(validator.boundaryConstraints.count, 2, "2 boundary constraints")
    } catch {
        expect(false, "Failed to build Fibonacci AIR: \(error)")
    }
}

private func testValidatorFromExpressions() {
    do {
        let col = FrAIRExprBuilder.self
        let validator = try GPUAIRTraceValidatorEngine.fromExpressions(
            numColumns: 2,
            logTraceLength: 3,
            transitions: [
                col.next(0) - col.col(1),
                col.next(1) - (col.col(0) + col.col(1))
            ],
            boundaries: [(column: 0, row: 0, value: Fr.one)]
        )
        expectEqual(validator.numColumns, 2, "From expressions: 2 columns")
        expectEqual(validator.transitionConstraints.count, 2, "2 transitions")
        expectEqual(validator.boundaryConstraints.count, 1, "1 boundary")
    } catch {
        expect(false, "Failed to build from expressions: \(error)")
    }
}

private func testValidatorDimensions() {
    do {
        let col = FrAIRExprBuilder.self
        let validator = try GPUAIRTraceValidatorEngine.fromExpressions(
            numColumns: 2,
            logTraceLength: 3,
            transitions: [col.next(0) - col.col(1)]
        )

        // Wrong number of columns
        let badTrace: [[Fr]] = [[Fr.one]]
        let report = validator.validateTrace(badTrace)
        expect(!report.isValid, "Wrong column count should fail")
        expect(report.violations.count == 1, "One dimension violation")

        // Wrong row count
        let badTrace2: [[Fr]] = [
            [Fr.one, Fr.one],
            [Fr.one, Fr.one]
        ]
        let report2 = validator.validateTrace(badTrace2)
        expect(!report2.isValid, "Wrong row count should fail")
    } catch {
        expect(false, "Dimension test failed: \(error)")
    }
}

// MARK: - Transition Constraints

private func testFibonacciTransitionValid() {
    do {
        let air = try FrAIRStandardLibrary.fibonacci(logTraceLength: 4)
        let validator = GPUAIRTraceValidatorEngine.fromAIR(air)
        let trace = air.generateTrace()
        let (violations, evals) = validator.validateTransitionConstraints(trace)
        expect(violations.isEmpty, "Valid Fibonacci trace has no transition violations")
        expectEqual(evals, 30, "15 rows * 2 constraints = 30 evaluations")
    } catch {
        expect(false, "Fibonacci transition test failed: \(error)")
    }
}

private func testFibonacciTransitionInvalid() {
    do {
        let air = try FrAIRStandardLibrary.fibonacci(logTraceLength: 4)
        let validator = GPUAIRTraceValidatorEngine.fromAIR(air)
        var trace = air.generateTrace()
        // Corrupt row 5
        trace[0][5] = frFromInt(999)
        let (violations, _) = validator.validateTransitionConstraints(trace)
        expect(!violations.isEmpty, "Corrupted trace has violations")
        // Should have violations at rows near the corruption
        let violatedRows = Set(violations.map { $0.row })
        expect(violatedRows.contains(4) || violatedRows.contains(5),
               "Violation near corrupted row 5")
    } catch {
        expect(false, "Invalid Fibonacci test failed: \(error)")
    }
}

private func testCounterTransition() {
    do {
        let col = FrAIRExprBuilder.self
        // next(0) - col(0) - 1 = 0
        let validator = try GPUAIRTraceValidatorEngine.fromExpressions(
            numColumns: 1,
            logTraceLength: 3,
            transitions: [col.next(0) - col.col(0) - col.constant(1)]
        )
        let trace = generateCounterTrace(logN: 3)
        let (violations, evals) = validator.validateTransitionConstraints(trace)
        expect(violations.isEmpty, "Counter trace valid: no violations")
        expectEqual(evals, 7, "7 row pairs checked")
    } catch {
        expect(false, "Counter transition test failed: \(error)")
    }
}

private func testQuadraticTransition() {
    do {
        let col = FrAIRExprBuilder.self
        // next(0) - col(0) * col(0) = 0  (squaring sequence)
        let validator = try GPUAIRTraceValidatorEngine.fromExpressions(
            numColumns: 1,
            logTraceLength: 2,
            transitions: [col.next(0) - col.col(0) * col.col(0)]
        )
        // Build squaring trace: 2, 4, 16, 256
        let trace: [[Fr]] = [[frFromInt(2), frFromInt(4), frFromInt(16), frFromInt(256)]]
        let (violations, _) = validator.validateTransitionConstraints(trace)
        expect(violations.isEmpty, "Squaring trace valid")

        // Invalid squaring trace
        let badTrace: [[Fr]] = [[frFromInt(2), frFromInt(4), frFromInt(15), frFromInt(256)]]
        let (badViolations, _) = validator.validateTransitionConstraints(badTrace)
        expect(!badViolations.isEmpty, "Bad squaring trace has violations")
        expectEqual(badViolations[0].row, 1, "Violation at row 1 (4->15 != 4^2)")
    } catch {
        expect(false, "Quadratic transition test failed: \(error)")
    }
}

// MARK: - Boundary Constraints

private func testBoundaryValid() {
    do {
        let air = try FrAIRStandardLibrary.fibonacci(logTraceLength: 4)
        let validator = GPUAIRTraceValidatorEngine.fromAIR(air)
        let trace = air.generateTrace()
        let violations = validator.validateBoundaryConstraints(trace)
        expect(violations.isEmpty, "Valid trace passes boundary checks")
    } catch {
        expect(false, "Boundary valid test failed: \(error)")
    }
}

private func testBoundaryInvalid() {
    do {
        let col = FrAIRExprBuilder.self
        let validator = try GPUAIRTraceValidatorEngine.fromExpressions(
            numColumns: 1,
            logTraceLength: 2,
            transitions: [col.next(0) - col.col(0)],
            boundaries: [(column: 0, row: 0, value: frFromInt(42))]
        )
        // Trace starts with 0 instead of 42
        let trace: [[Fr]] = [[Fr.zero, Fr.zero, Fr.zero, Fr.zero]]
        let violations = validator.validateBoundaryConstraints(trace)
        expectEqual(violations.count, 1, "One boundary violation")
        expectEqual(violations[0].row, 0, "Violation at row 0")
    } catch {
        expect(false, "Boundary invalid test failed: \(error)")
    }
}

private func testMultipleBoundaries() {
    do {
        let col = FrAIRExprBuilder.self
        let validator = try GPUAIRTraceValidatorEngine.fromExpressions(
            numColumns: 2,
            logTraceLength: 2,
            transitions: [col.next(0) - col.col(0)],
            boundaries: [
                (column: 0, row: 0, value: frFromInt(10)),
                (column: 1, row: 0, value: frFromInt(20)),
                (column: 0, row: 3, value: frFromInt(10))
            ]
        )
        // col0 = [10, 10, 10, 10], col1 = [20, x, x, x]
        let trace: [[Fr]] = [
            [frFromInt(10), frFromInt(10), frFromInt(10), frFromInt(10)],
            [frFromInt(20), frFromInt(0), frFromInt(0), frFromInt(0)]
        ]
        let violations = validator.validateBoundaryConstraints(trace)
        expect(violations.isEmpty, "All 3 boundaries satisfied")
    } catch {
        expect(false, "Multiple boundaries test failed: \(error)")
    }
}

private func testBoundaryOutOfRange() {
    do {
        let col = FrAIRExprBuilder.self
        let validator = try GPUAIRTraceValidatorEngine.fromExpressions(
            numColumns: 1,
            logTraceLength: 2,
            transitions: [col.next(0) - col.col(0)],
            boundaries: [(column: 5, row: 0, value: Fr.one)]
        )
        let trace: [[Fr]] = [[Fr.one, Fr.one, Fr.one, Fr.one]]
        let violations = validator.validateBoundaryConstraints(trace)
        expectEqual(violations.count, 1, "Out-of-range column detected")
    } catch {
        expect(false, "Boundary OOB test failed: \(error)")
    }
}

// MARK: - Periodic Constraints

private func testPeriodicEveryOtherRow() {
    do {
        let col = FrAIRExprBuilder.self
        // Constraint: col(0) must equal 0 at even rows
        let expr = col.col(0)
        let compiled = try GPUAIRConstraintCompiler.compileExpression(expr, numColumns: 1)
        let periodic = FrPeriodicConstraint(
            constraint: compiled, period: 2, phase: 0, label: "zero at even rows"
        )

        let validator = GPUAIRTraceValidatorEngine(
            numColumns: 1,
            logTraceLength: 3,
            transitionConstraints: [],
            periodicConstraints: [periodic]
        )

        // Trace where even rows are 0, odd rows are nonzero
        var trace = [Fr](repeating: Fr.zero, count: 8)
        trace[1] = frFromInt(5)
        trace[3] = frFromInt(7)
        trace[5] = frFromInt(9)
        // row 7 doesn't matter (no next row to pair with at n-1)
        let (violations, evals) = validator.validatePeriodicConstraints([trace])
        expect(violations.isEmpty, "Even rows are zero: no periodic violations")
        // Period 2, phase 0: rows 0, 2, 4, 6 — but we check up to n-2=6
        expectEqual(evals, 4, "4 periodic evaluations (rows 0,2,4,6)")
    } catch {
        expect(false, "Periodic every other test failed: \(error)")
    }
}

private func testPeriodicWithPhase() {
    do {
        let col = FrAIRExprBuilder.self
        // Constraint: next(0) - col(0) - 2 = 0 (increment by 2) at phase 1
        let expr = col.next(0) - col.col(0) - col.constant(2)
        let compiled = try GPUAIRConstraintCompiler.compileExpression(expr, numColumns: 1)
        let periodic = FrPeriodicConstraint(
            constraint: compiled, period: 4, phase: 1, label: "double step at phase 1"
        )

        let validator = GPUAIRTraceValidatorEngine(
            numColumns: 1,
            logTraceLength: 3,
            transitionConstraints: [],
            periodicConstraints: [periodic]
        )

        // Trace: 0, 1, 3, 3, 3, 5, 3, 3
        // At row 1 (phase 1 of period 4): next(0)=3, col(0)=1, 3-1-2=0 valid
        // At row 5 (phase 1 of period 4): next(0)=3, col(0)=5, 3-5-2=-4 invalid
        let trace: [[Fr]] = [[
            frFromInt(0), frFromInt(1), frFromInt(3), frFromInt(3),
            frFromInt(3), frFromInt(5), frFromInt(3), frFromInt(3)
        ]]
        let (violations, evals) = validator.validatePeriodicConstraints(trace)
        expectEqual(evals, 2, "2 evaluations at rows 1, 5")
        expectEqual(violations.count, 1, "1 violation at row 5")
        if violations.count > 0 {
            expectEqual(violations[0].row, 5, "Violation at row 5")
        }
    } catch {
        expect(false, "Periodic with phase test failed: \(error)")
    }
}

private func testPeriodicNoViolations() {
    do {
        let col = FrAIRExprBuilder.self
        // Constraint: col(0) * col(0) - col(0) = 0 => col(0) is 0 or 1
        let expr = col.col(0) * col.col(0) - col.col(0)
        let compiled = try GPUAIRConstraintCompiler.compileExpression(expr, numColumns: 1)
        let periodic = FrPeriodicConstraint(
            constraint: compiled, period: 2, phase: 0, label: "boolean at even"
        )

        let validator = GPUAIRTraceValidatorEngine(
            numColumns: 1,
            logTraceLength: 2,
            transitionConstraints: [],
            periodicConstraints: [periodic]
        )

        // Trace: [0, 5, 1, 7] — even rows (0, 1) are boolean
        let trace: [[Fr]] = [[Fr.zero, frFromInt(5), Fr.one, frFromInt(7)]]
        let (violations, _) = validator.validatePeriodicConstraints(trace)
        expect(violations.isEmpty, "Boolean check passes at even rows")
    } catch {
        expect(false, "Periodic no violations test failed: \(error)")
    }
}

// MARK: - Degree Bound Checking

private func testDegreeBoundLinearTrace() {
    do {
        let col = FrAIRExprBuilder.self
        let validator = try GPUAIRTraceValidatorEngine.fromExpressions(
            numColumns: 1,
            logTraceLength: 3,
            transitions: [col.next(0) - col.col(0) - col.constant(1)]
        )
        // Linear counter: 0,1,2,...,7
        let trace = generateCounterTrace(logN: 3)
        let result = validator.checkDegreeBounds(trace)
        expect(result.withinBounds, "Linear trace within degree bounds")
        expect(result.violatingColumns.isEmpty, "No violating columns")
    } catch {
        expect(false, "Degree bound linear test failed: \(error)")
    }
}

private func testDegreeBoundQuadraticTrace() {
    do {
        let col = FrAIRExprBuilder.self
        let validator = try GPUAIRTraceValidatorEngine.fromExpressions(
            numColumns: 1,
            logTraceLength: 3,
            transitions: [col.next(0) - col.col(0)]
        )
        // Quadratic values: 0, 1, 4, 9, 16, 25, 36, 49
        let trace: [[Fr]] = [(0..<8).map { frFromInt(UInt64($0 * $0)) }]
        let result = validator.checkDegreeBounds(trace)
        expect(result.withinBounds, "Quadratic within default bounds (n-1=7)")
    } catch {
        expect(false, "Degree bound quadratic test failed: \(error)")
    }
}

private func testDegreeBoundExceeded() {
    do {
        let col = FrAIRExprBuilder.self
        let validator = try GPUAIRTraceValidatorEngine.fromExpressions(
            numColumns: 1,
            logTraceLength: 3,
            transitions: [col.next(0) - col.col(0)]
        )
        // Counter trace: degree 1 polynomial (values 0..7 => interpolates to degree ~7)
        let trace = generateCounterTrace(logN: 3)
        // Restrictive bound: max degree 0
        let result = validator.checkDegreeBounds(trace, maxDegree: 0)
        expect(!result.withinBounds, "Should exceed degree bound 0")
        expectEqual(result.violatingColumns.count, 1, "1 violating column")
    } catch {
        expect(false, "Degree bound exceeded test failed: \(error)")
    }
}

// MARK: - Composition Polynomial

private func testCompositionValid() {
    do {
        let air = try FrAIRStandardLibrary.fibonacci(logTraceLength: 4)
        let validator = GPUAIRTraceValidatorEngine.fromAIR(air)
        let trace = air.generateTrace()
        let alpha = frFromInt(13)
        let result = validator.validateCompositionPolynomial(trace, alpha: alpha)
        expect(result.isValid, "Valid Fibonacci: composition vanishes")
        expect(result.nonzeroRows.isEmpty, "No nonzero rows")
    } catch {
        expect(false, "Composition valid test failed: \(error)")
    }
}

private func testCompositionInvalid() {
    do {
        let air = try FrAIRStandardLibrary.fibonacci(logTraceLength: 4)
        let validator = GPUAIRTraceValidatorEngine.fromAIR(air)
        var trace = air.generateTrace()
        trace[0][3] = frFromInt(9999)
        let alpha = frFromInt(7)
        let result = validator.validateCompositionPolynomial(trace, alpha: alpha)
        expect(!result.isValid, "Corrupted trace: composition nonzero")
        expect(!result.nonzeroRows.isEmpty, "Has nonzero rows")
    } catch {
        expect(false, "Composition invalid test failed: \(error)")
    }
}

private func testValidatorCompositionMultipleConstraints() {
    do {
        let col = FrAIRExprBuilder.self
        let validator = try GPUAIRTraceValidatorEngine.fromExpressions(
            numColumns: 2,
            logTraceLength: 3,
            transitions: [
                col.next(0) - col.col(1),
                col.next(1) - (col.col(0) + col.col(1))
            ]
        )
        let trace = generateFibTrace(logN: 3)
        let alpha = frFromInt(42)
        let result = validator.validateCompositionPolynomial(trace, alpha: alpha)
        expect(result.isValid, "Valid Fibonacci with alpha=42: composition vanishes")
    } catch {
        expect(false, "Composition multiple constraints test failed: \(error)")
    }
}

// MARK: - Multi-Register Validation

private func testMultiRegisterValid() {
    do {
        let air = try FrAIRStandardLibrary.fibonacci(logTraceLength: 4)
        let validator = GPUAIRTraceValidatorEngine.fromAIR(air)
        let trace = air.generateTrace()
        let report = validator.validateMultiRegisterTrace(trace)
        expect(report.isValid, "Multi-register Fibonacci valid")
    } catch {
        expect(false, "Multi-register valid test failed: \(error)")
    }
}

private func testMultiRegisterCrossConstraint() {
    do {
        let col = FrAIRExprBuilder.self
        // 2 columns, transition: next(0) = col(0) + 1
        // Cross-register: col(1) - col(0) * col(0) = 0 (col1 = col0^2)
        let validator = try GPUAIRTraceValidatorEngine.fromExpressions(
            numColumns: 2,
            logTraceLength: 2,
            transitions: [col.next(0) - col.col(0) - col.constant(1)]
        )

        let crossExpr = col.col(1) - col.col(0) * col.col(0)
        let crossCompiled = try GPUAIRConstraintCompiler.compileExpression(
            crossExpr, numColumns: 2)

        // Trace: col0=[0,1,2,3], col1=[0,1,4,9]
        let trace: [[Fr]] = [
            [frFromInt(0), frFromInt(1), frFromInt(2), frFromInt(3)],
            [frFromInt(0), frFromInt(1), frFromInt(4), frFromInt(9)]
        ]
        let report = validator.validateMultiRegisterTrace(
            trace, crossRegisterConstraints: [crossCompiled])
        expect(report.isValid, "Cross-register constraint col1=col0^2 valid")
    } catch {
        expect(false, "Multi-register cross constraint test failed: \(error)")
    }
}

private func testMultiRegisterMixed() {
    do {
        let col = FrAIRExprBuilder.self
        let validator = try GPUAIRTraceValidatorEngine.fromExpressions(
            numColumns: 2,
            logTraceLength: 2,
            transitions: [col.next(0) - col.col(0) - col.constant(1)],
            boundaries: [(column: 0, row: 0, value: Fr.zero)]
        )

        let crossExpr = col.col(1) - col.col(0) * col.col(0)
        let crossCompiled = try GPUAIRConstraintCompiler.compileExpression(
            crossExpr, numColumns: 2)

        // Intentionally wrong: col1 != col0^2 at row 2
        let trace: [[Fr]] = [
            [frFromInt(0), frFromInt(1), frFromInt(2), frFromInt(3)],
            [frFromInt(0), frFromInt(1), frFromInt(5), frFromInt(9)]  // row 2: 5 != 4
        ]
        let report = validator.validateMultiRegisterTrace(
            trace, crossRegisterConstraints: [crossCompiled])
        expect(!report.isValid, "Mixed violations detected")
        // Should have cross-register violation at row 2
        let crossViolations = report.violations.filter {
            $0.description.contains("Cross-register")
        }
        expect(!crossViolations.isEmpty, "Has cross-register violation")
    } catch {
        expect(false, "Multi-register mixed test failed: \(error)")
    }
}

// MARK: - Full Validation Report

private func testFullReportValid() {
    do {
        let air = try FrAIRStandardLibrary.fibonacci(logTraceLength: 4)
        let validator = GPUAIRTraceValidatorEngine.fromAIR(air)
        let trace = air.generateTrace()
        let report = validator.validateTrace(trace)
        expect(report.isValid, "Full report: valid Fibonacci")
        expect(report.violations.isEmpty, "No violations")
        expectEqual(report.numTransitionConstraints, 2, "2 transition constraints")
        expectEqual(report.numBoundaryConstraints, 2, "2 boundary constraints")
        expect(report.totalEvaluations > 0, "Nonzero evaluations")
        expect(report.validationTimeSeconds >= 0, "Non-negative time")
    } catch {
        expect(false, "Full report valid test failed: \(error)")
    }
}

private func testFullReportInvalid() {
    do {
        let air = try FrAIRStandardLibrary.fibonacci(logTraceLength: 4)
        let validator = GPUAIRTraceValidatorEngine.fromAIR(air)
        var trace = air.generateTrace()
        // Corrupt boundary: change initial value
        trace[0][0] = frFromInt(999)
        let report = validator.validateTrace(trace)
        expect(!report.isValid, "Full report: corrupted trace invalid")
        expect(!report.violations.isEmpty, "Has violations")
    } catch {
        expect(false, "Full report invalid test failed: \(error)")
    }
}

private func testFullReportSummary() {
    do {
        let air = try FrAIRStandardLibrary.fibonacci(logTraceLength: 3)
        let validator = GPUAIRTraceValidatorEngine.fromAIR(air)
        let trace = air.generateTrace()
        let report = validator.validateTrace(trace)
        let summary = report.summary
        expect(summary.contains("VALID"), "Summary says VALID")
        expect(summary.contains("transition"), "Summary mentions transitions")
        expect(summary.contains("boundary"), "Summary mentions boundaries")
    } catch {
        expect(false, "Summary test failed: \(error)")
    }
}

// MARK: - Batch Evaluation

private func testBatchEvalAllZero() {
    do {
        let air = try FrAIRStandardLibrary.fibonacci(logTraceLength: 4)
        let validator = GPUAIRTraceValidatorEngine.fromAIR(air)
        let trace = air.generateTrace()
        let evals = validator.batchEvaluateTransitions(trace)
        expectEqual(evals.count, 2, "2 constraints")
        for ci in 0..<2 {
            for row in 0..<15 {
                expect(evals[ci][row].isZero,
                       "Constraint \(ci) at row \(row) is zero")
            }
        }
    } catch {
        expect(false, "Batch eval all zero test failed: \(error)")
    }
}

private func testBatchEvalWithViolations() {
    do {
        let air = try FrAIRStandardLibrary.fibonacci(logTraceLength: 3)
        let validator = GPUAIRTraceValidatorEngine.fromAIR(air)
        var trace = air.generateTrace()
        trace[1][3] = frFromInt(12345)
        let evals = validator.batchEvaluateTransitions(trace)
        // At least one nonzero evaluation around corrupted row
        var foundNonzero = false
        for ci in 0..<evals.count {
            for row in 0..<evals[ci].count {
                if !evals[ci][row].isZero {
                    foundNonzero = true
                }
            }
        }
        expect(foundNonzero, "Corrupted trace has nonzero batch evaluations")
    } catch {
        expect(false, "Batch eval with violations test failed: \(error)")
    }
}

// MARK: - Degree Analysis

private func testValidatorDegreeAnalysisLinear() {
    do {
        let col = FrAIRExprBuilder.self
        let validator = try GPUAIRTraceValidatorEngine.fromExpressions(
            numColumns: 2,
            logTraceLength: 4,
            transitions: [col.next(0) - col.col(1)]
        )
        let analysis = validator.analyzeDegrees()
        expectEqual(analysis.maxTransitionDegree, 1, "Linear max degree = 1")
        expectEqual(analysis.compositionDegreeBound, 16, "Bound = 1 * 16")
        expectEqual(analysis.numQuotientChunks, 1, "1 chunk")
    } catch {
        expect(false, "Degree analysis linear test failed: \(error)")
    }
}

private func testValidatorDegreeAnalysisMixed() {
    do {
        let col = FrAIRExprBuilder.self
        let validator = try GPUAIRTraceValidatorEngine.fromExpressions(
            numColumns: 2,
            logTraceLength: 3,
            transitions: [
                col.next(0) - col.col(1),                    // degree 1
                col.col(0) * col.col(1)                      // degree 2
            ]
        )
        let analysis = validator.analyzeDegrees()
        expectEqual(analysis.maxTransitionDegree, 2, "Mixed max degree = 2")
        expectEqual(analysis.numQuotientChunks, 2, "2 chunks")
    } catch {
        expect(false, "Degree analysis mixed test failed: \(error)")
    }
}

// MARK: - Edge Cases

private func testSingleRowConstraint() {
    do {
        let col = FrAIRExprBuilder.self
        let validator = try GPUAIRTraceValidatorEngine.fromExpressions(
            numColumns: 1,
            logTraceLength: 1,
            transitions: [col.next(0) - col.col(0) - col.constant(1)]
        )
        // 2-element trace: [0, 1]
        let trace: [[Fr]] = [[Fr.zero, Fr.one]]
        let report = validator.validateTrace(trace)
        expect(report.isValid, "2-element counter trace valid")
        expectEqual(report.totalEvaluations, 1, "1 transition evaluation")
    } catch {
        expect(false, "Single row constraint test failed: \(error)")
    }
}

private func testCountViolations() {
    do {
        let air = try FrAIRStandardLibrary.fibonacci(logTraceLength: 3)
        let validator = GPUAIRTraceValidatorEngine.fromAIR(air)

        let trace = air.generateTrace()
        expectEqual(validator.countViolations(trace), 0, "Valid trace: 0 violations")

        var bad = trace
        bad[0][2] = frFromInt(777)
        let count = validator.countViolations(bad)
        expect(count > 0, "Corrupted trace: >0 violations")
    } catch {
        expect(false, "Count violations test failed: \(error)")
    }
}

private func testEvaluateAtSpecificRow() {
    do {
        let air = try FrAIRStandardLibrary.fibonacci(logTraceLength: 4)
        let validator = GPUAIRTraceValidatorEngine.fromAIR(air)
        let trace = air.generateTrace()

        // Valid row: should return nil
        let result0 = validator.evaluateConstraintAtRow(constraintIndex: 0, trace: trace, row: 5)
        expect(result0 == nil, "Valid row returns nil")

        // Corrupt and check
        var bad = trace
        bad[0][6] = frFromInt(12345)
        let result1 = validator.evaluateConstraintAtRow(constraintIndex: 0, trace: bad, row: 5)
        // May or may not violate at row 5 depending on which constraint, but row 5->6 is bad
        let result2 = validator.evaluateConstraintAtRow(constraintIndex: 0, trace: bad, row: 6)
        // At least one of these should be nonzero
        let anyViolation = (result1 != nil) || (result2 != nil)
        expect(anyViolation, "Corrupted row detected by evaluateConstraintAtRow")

        // Out of range constraint index
        let resultOOB = validator.evaluateConstraintAtRow(constraintIndex: 99, trace: trace, row: 0)
        expect(resultOOB == nil, "Out of range constraint returns nil")

        // Out of range row
        let resultRowOOB = validator.evaluateConstraintAtRow(constraintIndex: 0, trace: trace, row: 999)
        expect(resultRowOOB == nil, "Out of range row returns nil")
    } catch {
        expect(false, "Evaluate at specific row test failed: \(error)")
    }
}
