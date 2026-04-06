// GPU STARK Deep Composition Tests — validates GPU-accelerated DEEP polynomial engine
//
// Tests:
//   - OOD sampling produces non-trivial points
//   - OOD evaluation correctness (trace + composition at zeta)
//   - Single DEEP quotient computation
//   - Batch DEEP quotient construction (trace + next-row + composition)
//   - Full DEEP composition with alpha mixing
//   - FRI preparation (iNTT + degree check)
//   - Verifier-side DEEP reconstruction
//   - Point evaluation of DEEP at arbitrary query
//   - Composition polynomial splitting
//   - Segment LDE evaluation round-trip
//   - End-to-end DEEP + FRI pipeline
//   - Invalid input rejection
//   - Domain point computation

import zkMetal
import Foundation

public func runGPUSTARKDeepCompositionTests() {
    suite("GPU STARK Deep Composition — OOD Sampling")
    testOODSampling()

    suite("GPU STARK Deep Composition — OOD Evaluation")
    testOODEvaluation()

    suite("GPU STARK Deep Composition — Single Quotient")
    testSingleQuotient()

    suite("GPU STARK Deep Composition — Batch Quotients")
    testBatchQuotients()

    suite("GPU STARK Deep Composition — Full Composition")
    testFullComposition()

    suite("GPU STARK Deep Composition — FRI Preparation")
    testFRIPreparation()

    suite("GPU STARK Deep Composition — Verifier DEEP Check")
    testVerifierDEEPCheck()

    suite("GPU STARK Deep Composition — Evaluate DEEP At Point")
    testEvaluateDEEPAtPoint()

    suite("GPU STARK Deep Composition — Composition Splitting")
    testCompositionSplitting()

    suite("GPU STARK Deep Composition — Segment LDE Evaluation")
    testSegmentLDEEvaluation()

    suite("GPU STARK Deep Composition — End-to-End Pipeline")
    testEndToEndPipeline()

    suite("GPU STARK Deep Composition — Invalid Input Rejection")
    testDEEPInvalidInput()

    suite("GPU STARK Deep Composition — Domain Points")
    testDomainPoints()

    suite("GPU STARK Deep Composition — Effective Degree")
    testEffectiveDegree()
}

// MARK: - Helper: Build Simple Trace

/// Build a simple trace with known polynomial coefficients.
/// Returns (coefficients per column, LDE evaluations per column).
private func buildSimpleTrace(
    engine: GPUSTARKDeepCompositionEngine,
    config: DEEPCompositionConfig
) throws -> (coeffs: [[Fr]], ldeColumns: [[Fr]]) {
    let n = config.traceLen
    let logN = config.logTraceLen

    // Create polynomial coefficients for each column: col_i(x) = (i+1) + (i+2)*x + ...
    var allCoeffs = [[Fr]]()
    for c in 0..<config.numTraceColumns {
        var coeffs = [Fr](repeating: Fr.zero, count: n)
        for j in 0..<n {
            coeffs[j] = frFromInt(UInt64(c * n + j + 1))
        }
        allCoeffs.append(coeffs)
    }

    // Evaluate on trace domain via NTT to get trace evaluations,
    // then LDE to get coset evaluations.
    let m = config.ldeDomainSize
    let logM = config.logLDEDomainSize
    var ldeColumns = [[Fr]]()

    for coeffs in allCoeffs {
        // Zero-pad + coset shift + NTT
        var padded = [Fr](repeating: Fr.zero, count: m)
        var gPow = Fr.one
        for i in 0..<n {
            padded[i] = frMul(coeffs[i], gPow)
            gPow = frMul(gPow, config.cosetShift)
        }
        let lde = NTTEngine.cpuNTT(padded, logN: logM)
        ldeColumns.append(lde)
    }

    return (allCoeffs, ldeColumns)
}

// MARK: - OOD Sampling

private func testOODSampling() {
    do {
        let engine = try GPUSTARKDeepCompositionEngine()
        let config = DEEPCompositionConfig(
            logTraceLen: 3, blowupFactor: 4, numTraceColumns: 2)

        // Different commitment hashes should produce different OOD points
        let zeta1 = engine.sampleOODPoint(commitmentHash: frFromInt(42), config: config)
        let zeta2 = engine.sampleOODPoint(commitmentHash: frFromInt(43), config: config)

        expect(!frEqual(zeta1, Fr.zero), "OOD point 1 should be non-zero")
        expect(!frEqual(zeta2, Fr.zero), "OOD point 2 should be non-zero")
        expect(!frEqual(zeta1, zeta2), "Different hashes produce different OOD points")

        // Same hash should produce same point (deterministic)
        let zeta1b = engine.sampleOODPoint(commitmentHash: frFromInt(42), config: config)
        expect(frEqual(zeta1, zeta1b), "Same hash produces same OOD point")

    } catch {
        expect(false, "OOD sampling test failed: \(error)")
    }
}

// MARK: - OOD Evaluation

private func testOODEvaluation() {
    do {
        let engine = try GPUSTARKDeepCompositionEngine()
        let config = DEEPCompositionConfig(
            logTraceLen: 3, blowupFactor: 4, numTraceColumns: 2,
            numCompositionSegments: 1)

        let (traceCoeffs, _) = try buildSimpleTrace(engine: engine, config: config)

        // Simple composition: constant polynomial = 99
        let compCoeffs: [[Fr]] = [[frFromInt(99)]]

        let zeta = frFromInt(7)
        let oodFrame = try engine.evaluateOOD(
            traceCoeffs: traceCoeffs,
            compositionCoeffs: compCoeffs,
            zeta: zeta,
            config: config
        )

        // Verify trace evals match manual Horner evaluation
        for c in 0..<config.numTraceColumns {
            let expected = engine.evaluatePolynomial(traceCoeffs[c], at: zeta)
            expect(frEqual(oodFrame.traceEvals[c], expected),
                   "OOD trace eval col \(c) matches Horner")
        }

        // Verify next-row evals at zeta * omega
        let omega = frRootOfUnity(logN: config.logTraceLen)
        let zetaNext = frMul(zeta, omega)
        for c in 0..<config.numTraceColumns {
            let expected = engine.evaluatePolynomial(traceCoeffs[c], at: zetaNext)
            expect(frEqual(oodFrame.traceNextEvals[c], expected),
                   "OOD trace next eval col \(c) matches Horner")
        }

        // Composition eval at zeta = 99 (constant)
        expect(frEqual(oodFrame.compositionEvals[0], frFromInt(99)),
               "Composition eval = 99 (constant poly)")

    } catch {
        expect(false, "OOD evaluation test failed: \(error)")
    }
}

// MARK: - Single Quotient

private func testSingleQuotient() {
    do {
        let engine = try GPUSTARKDeepCompositionEngine()

        // f(x) = 3 + 2x + x^2
        // f(5) = 3 + 10 + 25 = 38
        // Q(x) = (f(x) - 38) / (x - 5) = (x^2 + 2x + 3 - 38) / (x - 5)
        //       = (x^2 + 2x - 35) / (x - 5) = (x + 7)(x - 5) / (x - 5) = x + 7
        let oodPoint = frFromInt(5)
        let oodEval = frFromInt(38)

        // Evaluate at a few domain points
        let domainPoints = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(10)]
        let columnEvals: [Fr] = domainPoints.map { pt in
            // f(pt) = pt^2 + 2*pt + 3
            frAdd(frAdd(frMul(pt, pt), frMul(frFromInt(2), pt)), frFromInt(3))
        }

        let quotient = try engine.computeQuotient(
            columnEvals: columnEvals,
            oodEval: oodEval,
            oodPoint: oodPoint,
            domainPoints: domainPoints,
            label: "test_quotient"
        )

        expectEqual(quotient.evaluations.count, 4, "Quotient has 4 evaluations")
        expectEqual(quotient.label, "test_quotient", "Label preserved")

        // Q(1) = 1 + 7 = 8
        expect(frEqual(quotient.evaluations[0], frFromInt(8)), "Q(1) = 8")
        // Q(2) = 2 + 7 = 9
        expect(frEqual(quotient.evaluations[1], frFromInt(9)), "Q(2) = 9")
        // Q(3) = 3 + 7 = 10
        expect(frEqual(quotient.evaluations[2], frFromInt(10)), "Q(3) = 10")
        // Q(10) = 10 + 7 = 17
        expect(frEqual(quotient.evaluations[3], frFromInt(17)), "Q(10) = 17")

    } catch {
        expect(false, "Single quotient test failed: \(error)")
    }
}

// MARK: - Batch Quotients

private func testBatchQuotients() {
    do {
        let engine = try GPUSTARKDeepCompositionEngine()
        let config = DEEPCompositionConfig(
            logTraceLen: 3, blowupFactor: 4, numTraceColumns: 2,
            numCompositionSegments: 1)

        let (traceCoeffs, ldeColumns) = try buildSimpleTrace(engine: engine, config: config)

        // Simple composition segment: constant = 42
        let compCoeffs: [[Fr]] = [[frFromInt(42)]]
        let m = config.ldeDomainSize
        let compLDE: [[Fr]] = [[Fr](repeating: frFromInt(42), count: m)]

        let zeta = frFromInt(7)
        let oodFrame = try engine.evaluateOOD(
            traceCoeffs: traceCoeffs,
            compositionCoeffs: compCoeffs,
            zeta: zeta,
            config: config
        )

        let quotients = try engine.computeAllQuotients(
            traceLDEColumns: ldeColumns,
            compositionLDESegments: compLDE,
            oodFrame: oodFrame,
            config: config
        )

        // Expected: 2 trace-at-zeta + 2 trace-at-zeta*omega + 1 composition = 5
        let expected = 2 * config.numTraceColumns + config.numCompositionSegments
        expectEqual(quotients.count, expected,
                    "Total quotients = 2*numTrace + numComp = \(expected)")

        // All quotients should have LDE domain size evaluations
        for (i, q) in quotients.enumerated() {
            expectEqual(q.evaluations.count, m,
                        "Quotient \(i) (\(q.label)) has \(m) evaluations")
        }

        // Check labels
        expect(quotients[0].label == "trace_col_0_at_zeta", "First quotient label correct")
        expect(quotients[2].label == "trace_col_0_at_zeta_omega", "Third quotient label correct")
        expect(quotients[4].label == "composition_seg_0", "Last quotient label correct")

    } catch {
        expect(false, "Batch quotients test failed: \(error)")
    }
}

// MARK: - Full Composition

private func testFullComposition() {
    do {
        let engine = try GPUSTARKDeepCompositionEngine()
        let config = DEEPCompositionConfig(
            logTraceLen: 3, blowupFactor: 4, numTraceColumns: 2,
            numCompositionSegments: 1)

        let (traceCoeffs, ldeColumns) = try buildSimpleTrace(engine: engine, config: config)
        let m = config.ldeDomainSize

        let compCoeffs: [[Fr]] = [[frFromInt(42)]]
        let compLDE: [[Fr]] = [[Fr](repeating: frFromInt(42), count: m)]

        let zeta = frFromInt(7)
        let alpha = frFromInt(13)

        let oodFrame = try engine.evaluateOOD(
            traceCoeffs: traceCoeffs,
            compositionCoeffs: compCoeffs,
            zeta: zeta,
            config: config
        )

        let result = try engine.compose(
            traceLDEColumns: ldeColumns,
            compositionLDESegments: compLDE,
            oodFrame: oodFrame,
            alpha: alpha,
            config: config
        )

        // Check result dimensions
        expectEqual(result.composedEvaluations.count, m,
                    "Composed polynomial has LDE domain size evaluations")
        expectEqual(result.numQuotients, 5,
                    "5 quotients (2 trace*2 points + 1 composition)")
        expectEqual(result.alphas.count, 5, "5 alpha powers")

        // Alpha powers should be 1, alpha, alpha^2, alpha^3, alpha^4
        expect(frEqual(result.alphas[0], Fr.one), "alpha^0 = 1")
        expect(frEqual(result.alphas[1], alpha), "alpha^1 = alpha")
        expect(frEqual(result.alphas[2], frMul(alpha, alpha)), "alpha^2")

        // Composed evaluations should be non-trivial
        var allZero = true
        for val in result.composedEvaluations {
            if !frEqual(val, Fr.zero) { allZero = false; break }
        }
        expect(!allZero, "Composed polynomial is non-trivial")

        // Verify the composition is the alpha-weighted sum of individual quotients
        let idx = 3  // check at domain index 3
        var expectedVal = Fr.zero
        for (t, q) in result.quotients.enumerated() {
            expectedVal = frAdd(expectedVal, frMul(result.alphas[t], q.evaluations[idx]))
        }
        expect(frEqual(result.composedEvaluations[idx], expectedVal),
               "Composed[3] = sum of alpha^t * Q_t[3]")

    } catch {
        expect(false, "Full composition test failed: \(error)")
    }
}

// MARK: - FRI Preparation

private func testFRIPreparation() {
    do {
        let engine = try GPUSTARKDeepCompositionEngine()
        let config = DEEPCompositionConfig(
            logTraceLen: 3, blowupFactor: 4, numTraceColumns: 1,
            numCompositionSegments: 1)

        let (traceCoeffs, ldeColumns) = try buildSimpleTrace(engine: engine, config: config)
        let m = config.ldeDomainSize

        let compCoeffs: [[Fr]] = [[frFromInt(5)]]
        let compLDE: [[Fr]] = [[Fr](repeating: frFromInt(5), count: m)]

        let zeta = frFromInt(11)
        let alpha = frFromInt(3)

        let oodFrame = try engine.evaluateOOD(
            traceCoeffs: traceCoeffs,
            compositionCoeffs: compCoeffs,
            zeta: zeta,
            config: config
        )

        let result = try engine.compose(
            traceLDEColumns: ldeColumns,
            compositionLDESegments: compLDE,
            oodFrame: oodFrame,
            alpha: alpha,
            config: config
        )

        // Prepare for FRI without degree bound
        let friCoeffs = try engine.prepareForFRI(result: result)
        expectEqual(friCoeffs.count, m, "FRI coefficients have LDE domain size")

        // At least some coefficients should be non-zero
        var hasNonzero = false
        for c in friCoeffs {
            if !frEqual(c, Fr.zero) { hasNonzero = true; break }
        }
        expect(hasNonzero, "FRI coefficients are non-trivial")

    } catch {
        expect(false, "FRI preparation test failed: \(error)")
    }
}

// MARK: - Verifier DEEP Check

private func testVerifierDEEPCheck() {
    do {
        let engine = try GPUSTARKDeepCompositionEngine()

        let traceEvals: [Fr] = [frFromInt(10), frFromInt(20)]
        let traceNextEvals: [Fr] = [frFromInt(30), frFromInt(40)]
        let compEvals: [Fr] = [frFromInt(50)]
        let alpha = frFromInt(7)

        let oodFrame = OODEvaluationFrame(
            zeta: frFromInt(99),
            traceEvals: traceEvals,
            traceNextEvals: traceNextEvals,
            compositionEvals: compEvals
        )

        // Compute expected: sum alpha^i * eval_i
        // alpha^0*10 + alpha^1*20 + alpha^2*30 + alpha^3*40 + alpha^4*50
        var expected = Fr.zero
        var alphaPow = Fr.one
        let allEvals = traceEvals + traceNextEvals + compEvals
        for eval in allEvals {
            expected = frAdd(expected, frMul(alphaPow, eval))
            alphaPow = frMul(alphaPow, alpha)
        }

        // Should pass with correct value
        let pass = engine.verifyDEEPAtOOD(
            oodFrame: oodFrame, alpha: alpha, claimedValue: expected)
        expect(pass, "Verifier accepts correct DEEP value")

        // Should fail with wrong value
        let wrongValue = frAdd(expected, Fr.one)
        let fail = engine.verifyDEEPAtOOD(
            oodFrame: oodFrame, alpha: alpha, claimedValue: wrongValue)
        expect(!fail, "Verifier rejects incorrect DEEP value")

    } catch {
        expect(false, "Verifier DEEP check test failed: \(error)")
    }
}

// MARK: - Evaluate DEEP At Point

private func testEvaluateDEEPAtPoint() {
    do {
        let engine = try GPUSTARKDeepCompositionEngine()
        let config = DEEPCompositionConfig(
            logTraceLen: 3, blowupFactor: 4, numTraceColumns: 2,
            numCompositionSegments: 1)

        let (traceCoeffs, ldeColumns) = try buildSimpleTrace(engine: engine, config: config)
        let m = config.ldeDomainSize

        let compCoeffs: [[Fr]] = [[frFromInt(42)]]
        let compLDE: [[Fr]] = [[Fr](repeating: frFromInt(42), count: m)]

        let zeta = frFromInt(7)
        let alpha = frFromInt(13)

        let oodFrame = try engine.evaluateOOD(
            traceCoeffs: traceCoeffs,
            compositionCoeffs: compCoeffs,
            zeta: zeta,
            config: config
        )

        let result = try engine.compose(
            traceLDEColumns: ldeColumns,
            compositionLDESegments: compLDE,
            oodFrame: oodFrame,
            alpha: alpha,
            config: config
        )

        // Pick a domain point and verify evaluateDEEPAt matches the composed value
        let domainPoints = engine.computeDomainPoints(config: config)
        let testIdx = 5
        let testPoint = domainPoints[testIdx]

        // Gather trace evals at this domain point
        var traceEvalsAtPt = [Fr]()
        for col in ldeColumns {
            traceEvalsAtPt.append(col[testIdx])
        }
        let compEvalsAtPt = [compLDE[0][testIdx]]

        let deepVal = engine.evaluateDEEPAt(
            point: testPoint,
            traceEvalsAtPoint: traceEvalsAtPt,
            compositionEvalsAtPoint: compEvalsAtPt,
            oodFrame: oodFrame,
            alpha: alpha,
            config: config
        )

        expect(frEqual(deepVal, result.composedEvaluations[testIdx]),
               "evaluateDEEPAt matches composed evaluation at domain point")

    } catch {
        expect(false, "Evaluate DEEP at point test failed: \(error)")
    }
}

// MARK: - Composition Splitting

private func testCompositionSplitting() {
    do {
        let engine = try GPUSTARKDeepCompositionEngine()
        let traceLen = 8

        // Composition poly: coeffs = [1, 2, 3, ..., 16]
        var coeffs = [Fr](repeating: Fr.zero, count: 16)
        for i in 0..<16 {
            coeffs[i] = frFromInt(UInt64(i + 1))
        }

        let segments = engine.splitCompositionPoly(
            coeffs: coeffs, traceLen: traceLen, numSegments: 2)

        expectEqual(segments.count, 2, "Split into 2 segments")
        expectEqual(segments[0].count, traceLen, "Segment 0 has traceLen coefficients")
        expectEqual(segments[1].count, traceLen, "Segment 1 has traceLen coefficients")

        // Segment 0: [1, 2, 3, 4, 5, 6, 7, 8]
        expect(frEqual(segments[0][0], frFromInt(1)), "Seg0[0] = 1")
        expect(frEqual(segments[0][7], frFromInt(8)), "Seg0[7] = 8")

        // Segment 1: [9, 10, 11, 12, 13, 14, 15, 16]
        expect(frEqual(segments[1][0], frFromInt(9)), "Seg1[0] = 9")
        expect(frEqual(segments[1][7], frFromInt(16)), "Seg1[7] = 16")

        // Test with shorter coefficients (zero-padding)
        let shortCoeffs = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let shortSegs = engine.splitCompositionPoly(
            coeffs: shortCoeffs, traceLen: 4, numSegments: 2)
        expectEqual(shortSegs.count, 2, "2 segments from short coeffs")
        expect(frEqual(shortSegs[0][0], frFromInt(1)), "Short seg0[0] = 1")
        expect(frEqual(shortSegs[0][2], frFromInt(3)), "Short seg0[2] = 3")
        expect(frEqual(shortSegs[0][3], Fr.zero), "Short seg0[3] = 0 (padding)")
        expect(frEqual(shortSegs[1][0], Fr.zero), "Short seg1[0] = 0 (no data)")

    } catch {
        expect(false, "Composition splitting test failed: \(error)")
    }
}

// MARK: - Segment LDE Evaluation

private func testSegmentLDEEvaluation() {
    do {
        let engine = try GPUSTARKDeepCompositionEngine()
        let config = DEEPCompositionConfig(
            logTraceLen: 3, blowupFactor: 4, numTraceColumns: 1,
            numCompositionSegments: 2)
        let n = config.traceLen

        // Two segments with known coefficients
        var seg0 = [Fr](repeating: Fr.zero, count: n)
        var seg1 = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            seg0[i] = frFromInt(UInt64(i + 1))
            seg1[i] = frFromInt(UInt64(i + 10))
        }

        let ldeSegments = try engine.evaluateSegmentsOnLDE(
            segments: [seg0, seg1], config: config)

        expectEqual(ldeSegments.count, 2, "Two LDE segments")
        expectEqual(ldeSegments[0].count, config.ldeDomainSize,
                    "Segment 0 LDE has domain size evaluations")
        expectEqual(ldeSegments[1].count, config.ldeDomainSize,
                    "Segment 1 LDE has domain size evaluations")

        // LDE evaluations should be non-trivial
        var hasNonzero = false
        for val in ldeSegments[0] {
            if !frEqual(val, Fr.zero) { hasNonzero = true; break }
        }
        expect(hasNonzero, "Segment 0 LDE is non-trivial")

        // Different segments should produce different LDE evaluations
        expect(!frEqual(ldeSegments[0][0], ldeSegments[1][0]),
               "Different segments produce different LDE evaluations")

    } catch {
        expect(false, "Segment LDE evaluation test failed: \(error)")
    }
}

// MARK: - End-to-End Pipeline

private func testEndToEndPipeline() {
    do {
        let engine = try GPUSTARKDeepCompositionEngine()
        let config = DEEPCompositionConfig(
            logTraceLen: 3, blowupFactor: 4, numTraceColumns: 2,
            numCompositionSegments: 1)

        let (traceCoeffs, ldeColumns) = try buildSimpleTrace(engine: engine, config: config)
        let n = config.traceLen

        // Composition: constant poly = 7
        let compCoeffs: [[Fr]] = [[frFromInt(7)]]
        let m = config.ldeDomainSize
        let compLDE: [[Fr]] = [[Fr](repeating: frFromInt(7), count: m)]

        let zeta = frFromInt(11)
        let alpha = frFromInt(5)

        let (deepResult, friCoeffs) = try engine.deepComposeForFRI(
            traceLDEColumns: ldeColumns,
            traceCoeffs: traceCoeffs,
            compositionLDESegments: compLDE,
            compositionCoeffs: compCoeffs,
            zeta: zeta,
            alpha: alpha,
            config: config
        )

        // Deep result should be well-formed
        expectEqual(deepResult.composedEvaluations.count, m, "Composed has LDE domain size")
        expectEqual(deepResult.numQuotients, 5, "5 quotients total")

        // FRI coefficients should be well-formed
        expectEqual(friCoeffs.count, m, "FRI coefficients have LDE domain size")

        // The OOD frame should be consistent
        expect(frEqual(deepResult.oodFrame.zeta, zeta), "OOD zeta preserved")

        // Verify OOD trace evals match direct computation
        for c in 0..<config.numTraceColumns {
            let expected = engine.evaluatePolynomial(traceCoeffs[c], at: zeta)
            expect(frEqual(deepResult.oodFrame.traceEvals[c], expected),
                   "E2E: trace eval col \(c) matches")
        }

    } catch {
        expect(false, "End-to-end pipeline test failed: \(error)")
    }
}

// MARK: - Invalid Input Rejection

private func testDEEPInvalidInput() {
    do {
        let engine = try GPUSTARKDeepCompositionEngine()
        let config = DEEPCompositionConfig(
            logTraceLen: 3, blowupFactor: 4, numTraceColumns: 2,
            numCompositionSegments: 1)

        // Wrong number of trace columns for OOD eval
        let wrongTraceCoeffs = [[frFromInt(1)]]  // Only 1 column, need 2
        let compCoeffs: [[Fr]] = [[frFromInt(5)]]
        let zeta = frFromInt(7)

        do {
            _ = try engine.evaluateOOD(
                traceCoeffs: wrongTraceCoeffs,
                compositionCoeffs: compCoeffs,
                zeta: zeta,
                config: config
            )
            expect(false, "Should reject wrong trace column count in OOD eval")
        } catch {
            expect(true, "Correctly rejected wrong trace column count")
        }

        // Wrong number of composition segments for OOD eval
        let traceCoeffs = [[frFromInt(1)], [frFromInt(2)]]
        let wrongCompCoeffs: [[Fr]] = [[frFromInt(1)], [frFromInt(2)]]  // 2 segments, need 1

        do {
            _ = try engine.evaluateOOD(
                traceCoeffs: traceCoeffs,
                compositionCoeffs: wrongCompCoeffs,
                zeta: zeta,
                config: config
            )
            expect(false, "Should reject wrong composition segment count")
        } catch {
            expect(true, "Correctly rejected wrong composition segment count")
        }

        // Mismatched column/domain sizes for quotient
        do {
            _ = try engine.computeQuotient(
                columnEvals: [frFromInt(1), frFromInt(2)],
                oodEval: frFromInt(1),
                oodPoint: frFromInt(5),
                domainPoints: [frFromInt(1)],  // size mismatch
                label: "bad"
            )
            expect(false, "Should reject mismatched sizes in quotient")
        } catch {
            expect(true, "Correctly rejected mismatched sizes")
        }

        // Wrong number of trace LDE columns for computeAllQuotients
        let oodFrame = OODEvaluationFrame(
            zeta: zeta,
            traceEvals: [frFromInt(1), frFromInt(2)],
            traceNextEvals: [frFromInt(3), frFromInt(4)],
            compositionEvals: [frFromInt(5)]
        )

        do {
            _ = try engine.computeAllQuotients(
                traceLDEColumns: [[frFromInt(1)]],  // Only 1, need 2
                compositionLDESegments: [[frFromInt(5)]],
                oodFrame: oodFrame,
                config: config
            )
            expect(false, "Should reject wrong trace LDE column count")
        } catch {
            expect(true, "Correctly rejected wrong trace LDE column count")
        }

    } catch {
        expect(false, "Invalid input rejection test failed: \(error)")
    }
}

// MARK: - Domain Points

private func testDomainPoints() {
    do {
        let engine = try GPUSTARKDeepCompositionEngine()
        let config = DEEPCompositionConfig(
            logTraceLen: 3, blowupFactor: 4, numTraceColumns: 1)

        let points = engine.computeDomainPoints(config: config)

        expectEqual(points.count, config.ldeDomainSize,
                    "Domain has LDE domain size points")

        // First point should be cosetShift * omega^0 = cosetShift
        expect(frEqual(points[0], config.cosetShift),
               "First domain point = coset shift")

        // All points should be distinct
        var allDistinct = true
        for i in 0..<points.count {
            for j in (i+1)..<min(i+5, points.count) {
                if frEqual(points[i], points[j]) {
                    allDistinct = false
                    break
                }
            }
            if !allDistinct { break }
        }
        expect(allDistinct, "Domain points are distinct (spot-checked)")

    } catch {
        expect(false, "Domain points test failed: \(error)")
    }
}

// MARK: - Effective Degree

private func testEffectiveDegree() {
    do {
        let engine = try GPUSTARKDeepCompositionEngine()

        // Zero polynomial: degree 0
        let zero = [Fr.zero, Fr.zero, Fr.zero]
        expectEqual(engine.effectiveDegree(of: zero), 0, "Zero poly has degree 0")

        // Constant: degree 0
        let constant = [frFromInt(5), Fr.zero, Fr.zero]
        expectEqual(engine.effectiveDegree(of: constant), 0, "Constant has degree 0")

        // Linear: degree 1
        let linear = [frFromInt(1), frFromInt(2), Fr.zero]
        expectEqual(engine.effectiveDegree(of: linear), 1, "Linear has degree 1")

        // Quadratic: degree 2
        let quadratic = [frFromInt(1), frFromInt(2), frFromInt(3)]
        expectEqual(engine.effectiveDegree(of: quadratic), 2, "Quadratic has degree 2")

        // Polynomial evaluation test
        let coeffs = [frFromInt(3), frFromInt(2), frFromInt(1)]  // 3 + 2x + x^2
        let at0 = engine.evaluatePolynomial(coeffs, at: Fr.zero)
        expect(frEqual(at0, frFromInt(3)), "p(0) = 3")
        let at1 = engine.evaluatePolynomial(coeffs, at: Fr.one)
        expect(frEqual(at1, frFromInt(6)), "p(1) = 6")

        let empty = engine.evaluatePolynomial([], at: frFromInt(7))
        expect(frEqual(empty, Fr.zero), "Empty poly = 0")

    } catch {
        expect(false, "Effective degree test failed: \(error)")
    }
}
