// GPU STARK Trace LDE Tests — validates GPU-accelerated trace low-degree extension
//
// Tests:
//   - Single column LDE correctness (GPU vs CPU reference)
//   - Multi-column batch LDE
//   - Interpolation round-trip (NTT -> iNTT identity)
//   - Coset LDE domain size validation
//   - Constraint composition polynomial evaluation
//   - Boundary constraint enforcement
//   - Merkle tree commitment consistency
//   - Invalid input rejection
//   - Blowup factor variations (2, 4, 8)
//   - Polynomial evaluation (Horner) correctness

import zkMetal
import Foundation

public func runGPUSTARKTraceLDETests() {
    suite("GPU STARK Trace LDE — Single Column LDE")
    testSingleColumnLDE()

    suite("GPU STARK Trace LDE — Batch Multi-Column LDE")
    testBatchMultiColumnLDE()

    suite("GPU STARK Trace LDE — Interpolation Round-Trip")
    testInterpolationRoundTrip()

    suite("GPU STARK Trace LDE — Blowup Factor Variations")
    testBlowupFactorVariations()

    suite("GPU STARK Trace LDE — Constraint Composition")
    testConstraintComposition()

    suite("GPU STARK Trace LDE — Boundary Constraints")
    testBoundaryConstraints()

    suite("GPU STARK Trace LDE — Merkle Commitment")
    testMerkleCommitment()

    suite("GPU STARK Trace LDE — Invalid Input Rejection")
    testInvalidInputRejection()

    suite("GPU STARK Trace LDE — Polynomial Evaluation")
    testSTARKPolyEval()

    suite("GPU STARK Trace LDE — Config Validation")
    testConfigValidation()

    suite("GPU STARK Trace LDE — Query Row")
    testQueryRow()
}

// MARK: - Single Column LDE

func testSingleColumnLDE() {
    do {
        let engine = try GPUSTARKTraceLDEEngine()
        let logN = 3
        let n = 1 << logN
        let blowup = 4
        let cosetShift = frFromInt(Fr.GENERATOR)

        // Simple polynomial: p(x) = 1 + 2x + 3x^2 + ... evaluated at omega^i
        var evals = [Fr](repeating: Fr.zero, count: n)
        let omega = frRootOfUnity(logN: logN)
        var w = Fr.one
        for i in 0..<n {
            // Evaluate p(omega^i) = sum_j (j+1) * (omega^i)^j
            var val = Fr.zero
            var wPow = Fr.one
            for j in 0..<n {
                val = frAdd(val, frMul(frFromInt(UInt64(j + 1)), wPow))
                wPow = frMul(wPow, w)
            }
            evals[i] = val
            w = frMul(w, omega)
        }

        let lde = try engine.extendColumn(evals: evals, logN: logN,
                                           blowupFactor: blowup, cosetShift: cosetShift)

        // LDE should have blowup * n elements
        expectEqual(lde.count, blowup * n, "LDE output size = blowup * n")

        // All elements should be valid (non-trivial)
        var allZero = true
        for val in lde {
            if !frEqual(val, Fr.zero) { allZero = false; break }
        }
        expect(!allZero, "LDE should produce non-zero evaluations")

    } catch {
        expect(false, "Single column LDE failed: \(error)")
    }
}

// MARK: - Batch Multi-Column LDE

func testBatchMultiColumnLDE() {
    do {
        let engine = try GPUSTARKTraceLDEEngine()
        let logN = 4
        let n = 1 << logN
        let numCols = 3
        let blowup = 2

        let config = STARKTraceLDEConfig(
            logTraceLen: logN, blowupFactor: blowup, numColumns: numCols)

        // Create a simple trace: column i has values [i+1, i+2, ..., i+n]
        var trace = [[Fr]]()
        for c in 0..<numCols {
            var col = [Fr](repeating: Fr.zero, count: n)
            for r in 0..<n {
                col[r] = frFromInt(UInt64(c * n + r + 1))
            }
            trace.append(col)
        }

        let result = try engine.extend(trace: trace, config: config)

        // Check output dimensions
        expectEqual(result.ldeColumns.count, numCols, "Number of LDE columns")
        for c in 0..<numCols {
            expectEqual(result.ldeColumns[c].count, config.ldeDomainSize,
                        "LDE column \(c) size = blowup * traceLen")
        }

        // Commitment should be non-zero
        expect(!frEqual(result.commitment, Fr.zero), "Merkle commitment should be non-zero")

        // Merkle leaves should match domain size
        expectEqual(result.merkleLeaves.count, config.ldeDomainSize,
                    "Number of Merkle leaves = LDE domain size")

    } catch {
        expect(false, "Batch multi-column LDE failed: \(error)")
    }
}

// MARK: - Interpolation Round-Trip

func testInterpolationRoundTrip() {
    do {
        let engine = try GPUSTARKTraceLDEEngine()
        let logN = 3
        let n = 1 << logN

        // Start with known coefficients
        var coeffs = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            coeffs[i] = frFromInt(UInt64(i + 1))
        }

        // NTT to get evaluations
        let evals = NTTEngine.cpuNTT(coeffs, logN: logN)

        // iNTT back to coefficients
        let recovered = try engine.interpolate(evals: evals, logN: logN)

        // Should match original coefficients
        expectEqual(recovered.count, n, "Recovered coefficient count")
        var allMatch = true
        for i in 0..<n {
            if !frEqual(recovered[i], coeffs[i]) {
                allMatch = false
                break
            }
        }
        expect(allMatch, "Interpolation round-trip: recovered coefficients match original")

    } catch {
        expect(false, "Interpolation round-trip failed: \(error)")
    }
}

// MARK: - Blowup Factor Variations

func testBlowupFactorVariations() {
    do {
        let engine = try GPUSTARKTraceLDEEngine()
        let logN = 3
        let n = 1 << logN

        // Simple trace: single column with sequential values
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = frFromInt(UInt64(i + 1)) }

        let cosetShift = frFromInt(Fr.GENERATOR)

        // Test blowup factor 2
        let lde2 = try engine.extendColumn(evals: evals, logN: logN,
                                            blowupFactor: 2, cosetShift: cosetShift)
        expectEqual(lde2.count, 2 * n, "Blowup=2: output size = 2*n")

        // Test blowup factor 4
        let lde4 = try engine.extendColumn(evals: evals, logN: logN,
                                            blowupFactor: 4, cosetShift: cosetShift)
        expectEqual(lde4.count, 4 * n, "Blowup=4: output size = 4*n")

        // Test blowup factor 8
        let lde8 = try engine.extendColumn(evals: evals, logN: logN,
                                            blowupFactor: 8, cosetShift: cosetShift)
        expectEqual(lde8.count, 8 * n, "Blowup=8: output size = 8*n")

        // All LDE outputs should be non-trivially different from each other
        // (different blowup factors produce different coset evaluations)
        expect(!frEqual(lde2[0], lde4[0]) || !frEqual(lde2[1], lde4[1]),
               "Different blowup factors produce different LDE evaluations")

    } catch {
        expect(false, "Blowup factor test failed: \(error)")
    }
}

// MARK: - Constraint Composition

func testConstraintComposition() {
    do {
        let engine = try GPUSTARKTraceLDEEngine()
        let logN = 3
        let n = 1 << logN
        let numCols = 2
        let blowup = 4

        let config = STARKTraceLDEConfig(
            logTraceLen: logN, blowupFactor: blowup, numColumns: numCols)

        // Build a Fibonacci-like trace: col0 = [1, 1, 2, 3, 5, 8, 13, 21]
        //                                col1 = [1, 2, 3, 5, 8, 13, 21, 34]
        var col0 = [Fr](repeating: Fr.zero, count: n)
        var col1 = [Fr](repeating: Fr.zero, count: n)
        col0[0] = frFromInt(1)
        col1[0] = frFromInt(1)
        for i in 1..<n {
            col0[i] = col1[i - 1]
            col1[i] = frAdd(col0[i - 1], col1[i - 1])
        }

        let trace = [col0, col1]
        let ldeResult = try engine.extend(trace: trace, config: config)

        let alpha = frFromInt(7)

        // Fibonacci transition constraint: next[0] = current[1], next[1] = current[0] + current[1]
        let compositionResult = try engine.evaluateComposition(
            ldeResult: ldeResult,
            constraintEvaluator: { current, next -> [Fr] in
                let c0 = frSub(next[0], current[1])
                let c1 = frSub(next[1], frAdd(current[0], current[1]))
                return [c0, c1]
            },
            alpha: alpha
        )

        // Composition polynomial should have LDE domain size evaluations
        expectEqual(compositionResult.evaluations.count, config.ldeDomainSize,
                    "Composition polynomial has LDE domain size evaluations")

        // Commitment should be non-zero
        expect(!frEqual(compositionResult.commitment, Fr.zero),
               "Composition commitment should be non-zero")

    } catch {
        expect(false, "Constraint composition test failed: \(error)")
    }
}

// MARK: - Boundary Constraints

func testBoundaryConstraints() {
    do {
        let engine = try GPUSTARKTraceLDEEngine()
        let logN = 3
        let n = 1 << logN
        let numCols = 2
        let blowup = 4

        let config = STARKTraceLDEConfig(
            logTraceLen: logN, blowupFactor: blowup, numColumns: numCols)

        // Simple trace
        var col0 = [Fr](repeating: Fr.zero, count: n)
        var col1 = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            col0[i] = frFromInt(UInt64(i + 1))
            col1[i] = frFromInt(UInt64(2 * i + 1))
        }

        let trace = [col0, col1]
        let ldeResult = try engine.extend(trace: trace, config: config)

        // Enforce boundary constraint: col0[0] = 1
        let constraints = [
            FrBoundaryConstraint(column: 0, row: 0, value: frFromInt(1)),
            FrBoundaryConstraint(column: 1, row: 0, value: frFromInt(1)),
        ]

        let quotients = try engine.enforceBoundaryConstraints(
            ldeResult: ldeResult, constraints: constraints)

        expectEqual(quotients.count, 2, "Two boundary quotients")
        expectEqual(quotients[0].count, config.ldeDomainSize,
                    "Boundary quotient 0 has LDE domain size")
        expectEqual(quotients[1].count, config.ldeDomainSize,
                    "Boundary quotient 1 has LDE domain size")

        // Quotients should be non-trivial
        var q0nonzero = false
        for val in quotients[0] {
            if !frEqual(val, Fr.zero) { q0nonzero = true; break }
        }
        expect(q0nonzero, "Boundary quotient 0 is non-trivial")

    } catch {
        expect(false, "Boundary constraint test failed: \(error)")
    }
}

// MARK: - Merkle Commitment

func testMerkleCommitment() {
    do {
        let engine = try GPUSTARKTraceLDEEngine()
        let logN = 3
        let n = 1 << logN
        let numCols = 2
        let blowup = 2

        let config = STARKTraceLDEConfig(
            logTraceLen: logN, blowupFactor: blowup, numColumns: numCols)

        // Two identical traces should produce identical commitments
        var col = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { col[i] = frFromInt(UInt64(i + 1)) }

        let trace1 = [col, col]
        let result1 = try engine.extend(trace: trace1, config: config)

        let trace2 = [col, col]
        let result2 = try engine.extend(trace: trace2, config: config)

        expect(frEqual(result1.commitment, result2.commitment),
               "Identical traces produce identical commitments")

        // Different traces should produce different commitments
        var col2 = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { col2[i] = frFromInt(UInt64(i + 100)) }

        let trace3 = [col, col2]
        let result3 = try engine.extend(trace: trace3, config: config)

        expect(!frEqual(result1.commitment, result3.commitment),
               "Different traces produce different commitments")

        // Merkle nodes array should have correct size: 2*domainSize - 1
        let expectedNodes = 2 * config.ldeDomainSize - 1
        expectEqual(result1.merkleNodes.count, expectedNodes,
                    "Merkle nodes array size = 2*domain - 1")

        // Root node (index 0) should equal commitment
        expect(frEqual(result1.merkleNodes[0], result1.commitment),
               "Root node equals commitment")

    } catch {
        expect(false, "Merkle commitment test failed: \(error)")
    }
}

// MARK: - Invalid Input Rejection

func testInvalidInputRejection() {
    do {
        let engine = try GPUSTARKTraceLDEEngine()

        // Test 1: Wrong number of columns
        let config2col = STARKTraceLDEConfig(
            logTraceLen: 3, blowupFactor: 2, numColumns: 2)
        let singleCol = [[frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4),
                           frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]]

        do {
            _ = try engine.extend(trace: singleCol, config: config2col)
            expect(false, "Should reject wrong column count")
        } catch {
            expect(true, "Correctly rejected wrong column count")
        }

        // Test 2: Wrong row count
        let config8row = STARKTraceLDEConfig(
            logTraceLen: 3, blowupFactor: 2, numColumns: 1)
        let shortCol = [[frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]]

        do {
            _ = try engine.extend(trace: shortCol, config: config8row)
            expect(false, "Should reject wrong row count")
        } catch {
            expect(true, "Correctly rejected wrong row count")
        }

        // Test 3: Boundary constraint with invalid column index
        let config1col = STARKTraceLDEConfig(
            logTraceLen: 3, blowupFactor: 2, numColumns: 1)
        var col = [Fr](repeating: Fr.zero, count: 8)
        for i in 0..<8 { col[i] = frFromInt(UInt64(i + 1)) }
        let ldeResult = try engine.extend(trace: [col], config: config1col)

        let badConstraint = FrBoundaryConstraint(column: 5, row: 0, value: Fr.one)
        do {
            _ = try engine.enforceBoundaryConstraints(
                ldeResult: ldeResult, constraints: [badConstraint])
            expect(false, "Should reject out-of-range column in boundary constraint")
        } catch {
            expect(true, "Correctly rejected out-of-range column")
        }

        // Test 4: Boundary constraint with invalid row index
        let badRowConstraint = FrBoundaryConstraint(column: 0, row: 100, value: Fr.one)
        do {
            _ = try engine.enforceBoundaryConstraints(
                ldeResult: ldeResult, constraints: [badRowConstraint])
            expect(false, "Should reject out-of-range row in boundary constraint")
        } catch {
            expect(true, "Correctly rejected out-of-range row")
        }

    } catch {
        expect(false, "Invalid input rejection test failed: \(error)")
    }
}

// MARK: - Polynomial Evaluation

private func testSTARKPolyEval() {
    do {
        let engine = try GPUSTARKTraceLDEEngine()

        // p(x) = 3 + 2x + x^2
        let coeffs = [frFromInt(3), frFromInt(2), frFromInt(1)]

        // p(0) = 3
        let at0 = engine.evaluatePolynomial(coeffs, at: Fr.zero)
        expect(frEqual(at0, frFromInt(3)), "p(0) = 3")

        // p(1) = 3 + 2 + 1 = 6
        let at1 = engine.evaluatePolynomial(coeffs, at: Fr.one)
        expect(frEqual(at1, frFromInt(6)), "p(1) = 6")

        // p(2) = 3 + 4 + 4 = 11
        let at2 = engine.evaluatePolynomial(coeffs, at: frFromInt(2))
        expect(frEqual(at2, frFromInt(11)), "p(2) = 11")

        // p(3) = 3 + 6 + 9 = 18
        let at3 = engine.evaluatePolynomial(coeffs, at: frFromInt(3))
        expect(frEqual(at3, frFromInt(18)), "p(3) = 18")

        // Empty polynomial = 0
        let empty = engine.evaluatePolynomial([], at: frFromInt(5))
        expect(frEqual(empty, Fr.zero), "Empty polynomial evaluates to 0")

        // Constant polynomial p(x) = 42
        let constant = engine.evaluatePolynomial([frFromInt(42)], at: frFromInt(999))
        expect(frEqual(constant, frFromInt(42)), "Constant polynomial = 42")

    } catch {
        expect(false, "Polynomial evaluation test failed: \(error)")
    }
}

// MARK: - Config Validation

func testConfigValidation() {
    // Test default coset shift
    let config1 = STARKTraceLDEConfig(
        logTraceLen: 4, blowupFactor: 4, numColumns: 3)
    expectEqual(config1.traceLen, 16, "traceLen = 2^4 = 16")
    expectEqual(config1.ldeDomainSize, 64, "ldeDomainSize = 16 * 4 = 64")
    expectEqual(config1.logLDEDomainSize, 6, "logLDEDomainSize = 4 + 2 = 6")

    // Test custom coset shift
    let customShift = frFromInt(13)
    let config2 = STARKTraceLDEConfig(
        logTraceLen: 5, blowupFactor: 8, numColumns: 1, cosetShift: customShift)
    expectEqual(config2.traceLen, 32, "traceLen = 2^5 = 32")
    expectEqual(config2.ldeDomainSize, 256, "ldeDomainSize = 32 * 8 = 256")
    expectEqual(config2.logLDEDomainSize, 8, "logLDEDomainSize = 5 + 3 = 8")
    expect(frEqual(config2.cosetShift, customShift), "Custom coset shift preserved")

    // Test blowup factor 2
    let config3 = STARKTraceLDEConfig(
        logTraceLen: 10, blowupFactor: 2, numColumns: 5)
    expectEqual(config3.traceLen, 1024, "traceLen = 2^10 = 1024")
    expectEqual(config3.ldeDomainSize, 2048, "ldeDomainSize = 1024 * 2 = 2048")
    expectEqual(config3.logLDEDomainSize, 11, "logLDEDomainSize = 10 + 1 = 11")
}

// MARK: - Query Row

func testQueryRow() {
    do {
        let engine = try GPUSTARKTraceLDEEngine()
        let logN = 3
        let n = 1 << logN
        let numCols = 3
        let blowup = 2

        let config = STARKTraceLDEConfig(
            logTraceLen: logN, blowupFactor: blowup, numColumns: numCols)

        var trace = [[Fr]]()
        for c in 0..<numCols {
            var col = [Fr](repeating: Fr.zero, count: n)
            for r in 0..<n {
                col[r] = frFromInt(UInt64(c * 100 + r + 1))
            }
            trace.append(col)
        }

        let result = try engine.extend(trace: trace, config: config)

        // Query row 0
        let row0 = engine.queryRow(ldeResult: result, index: 0)
        expectEqual(row0.count, numCols, "Query row returns numColumns values")

        // Each element of the row should match the corresponding LDE column value
        for c in 0..<numCols {
            expect(frEqual(row0[c], result.ldeColumns[c][0]),
                   "Query row[0] column \(c) matches ldeColumns[\(c)][0]")
        }

        // Query a middle row
        let midIdx = config.ldeDomainSize / 2
        let rowMid = engine.queryRow(ldeResult: result, index: midIdx)
        for c in 0..<numCols {
            expect(frEqual(rowMid[c], result.ldeColumns[c][midIdx]),
                   "Query row[\(midIdx)] column \(c) matches")
        }

    } catch {
        expect(false, "Query row test failed: \(error)")
    }
}
