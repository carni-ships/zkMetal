// GPU AIR Composition Engine Tests — validates composition polynomial construction,
// coset domain evaluation, chunk splitting, multi-trace support, deep composition,
// zerofier computation, boundary quotients, parallel evaluation, and degree analysis.

import Foundation
import zkMetal

public func runGPUAIRCompositionTests() {
    suite("GPUAIRComposition - Domain")
    testCompositionDomainBasic()
    testCompositionDomainElements()
    testCompositionDomainEvaluation()
    testCompositionDomainBlowup()

    suite("GPUAIRComposition - Engine Construction")
    testEngineFromConstraints()
    testEngineFromAIR()
    testEngineDimensions()
    testEngineConstraintGroups()

    suite("GPUAIRComposition - Single Constraint")
    testSingleConstraintComposition()
    testSingleConstraintChunks()
    testSingleConstraintValidTrace()

    suite("GPUAIRComposition - Multiple Constraints")
    testMultipleConstraintComposition()
    testMultipleConstraintAlphaPowers()
    testMultipleConstraintDegrees()

    suite("GPUAIRComposition - Zerofier")
    testZerofierOnDomain()
    testZerofierOffDomain()
    testZerofierOnCoset()
    testBoundaryZerofier()

    suite("GPUAIRComposition - Chunk Splitting")
    testChunkSplitSingle()
    testChunkSplitMultiple()
    testChunkDegreeBounds()

    suite("GPUAIRComposition - Multi-Trace")
    testAuxiliaryConstraintEval()
    testMultiTraceComposition()
    testMultiTraceResult()

    suite("GPUAIRComposition - Deep Composition")
    testDeepCompositionCoeffs()
    testDeepCompositionBasic()

    suite("GPUAIRComposition - Parallel")
    testParallelComposeMatchesSequential()
    testParallelComposeBlockSize()

    suite("GPUAIRComposition - Boundary Quotients")
    testBoundaryQuotientSingle()
    testBoundaryQuotientMultiple()

    suite("GPUAIRComposition - Degree Analysis")
    testAIRCompDegreeAnalysisLinear()
    testAIRCompDegreeAnalysisQuadratic()
    testDegreeBoundComputation()

    suite("GPUAIRComposition - Verification")
    testVerifyValidTrace()
    testVerifyInvalidTrace()

    suite("GPUAIRComposition - Fibonacci AIR")
    testFibonacciComposition()
    testFibonacciChunks()

    suite("GPUAIRComposition - Edge Cases")
    testSingleRowTrace()
    testLargeAlpha()
    testZeroAlpha()
    testComposeAll()
}

// MARK: - Helpers

private let logN = 4
private let traceLen = 1 << logN

private func makeFibTrace() -> [[Fr]] {
    var colA = [Fr](repeating: Fr.zero, count: traceLen)
    var colB = [Fr](repeating: Fr.zero, count: traceLen)
    colA[0] = Fr.one
    colB[0] = Fr.one
    for i in 1..<traceLen {
        colA[i] = colB[i - 1]
        colB[i] = frAdd(colA[i - 1], colB[i - 1])
    }
    return [colA, colB]
}

private func makeCounterTrace() -> [[Fr]] {
    var col = [Fr](repeating: Fr.zero, count: traceLen)
    for i in 0..<traceLen {
        col[i] = frFromInt(UInt64(i))
    }
    return [col]
}

private func compileFibConstraints() -> [CompiledFrConstraint] {
    let c1 = try! GPUAIRConstraintCompiler.compileExpression(
        FrAIRExprBuilder.next(0) - FrAIRExprBuilder.col(1),
        numColumns: 2)
    let c2 = try! GPUAIRConstraintCompiler.compileExpression(
        FrAIRExprBuilder.next(1) - (FrAIRExprBuilder.col(0) + FrAIRExprBuilder.col(1)),
        numColumns: 2)
    return [c1, c2]
}

private func compileCounterConstraint() -> [CompiledFrConstraint] {
    let c = try! GPUAIRConstraintCompiler.compileExpression(
        FrAIRExprBuilder.next(0) - FrAIRExprBuilder.col(0) - FrAIRExprBuilder.constant(1),
        numColumns: 1)
    return [c]
}

private func defaultDomain() -> CompositionDomain {
    CompositionDomain(logN: logN, cosetShift: frFromInt(7), blowupFactor: 4)
}

private func makeFibEngine() -> GPUAIRCompositionEngine {
    GPUAIRCompositionEngine(
        numMainColumns: 2,
        logTraceLength: logN,
        transitionConstraints: compileFibConstraints(),
        boundaryConstraints: [
            (column: 0, row: 0, value: Fr.one),
            (column: 1, row: 0, value: Fr.one)
        ])
}

// MARK: - Domain Tests

private func testCompositionDomainBasic() {
    let domain = CompositionDomain(logN: 4, cosetShift: frFromInt(5), blowupFactor: 4)
    expectEqual(domain.logN, 4, "logN is 4")
    expectEqual(domain.size, 16, "size is 16")
    expectEqual(domain.blowupFactor, 4, "blowupFactor is 4")
    expectEqual(domain.evaluationDomainSize, 64, "eval domain is 64")
    expectEqual(domain.logBlowup, 2, "logBlowup is 2")
    expectEqual(domain.logEvaluationDomainSize, 6, "logEvalDomainSize is 6")
}

private func testCompositionDomainElements() {
    let domain = CompositionDomain(logN: 3, cosetShift: frFromInt(5), blowupFactor: 2)
    let elements = domain.allElements()
    expectEqual(elements.count, 8, "8 elements in domain")
    // First element should be the coset shift
    expect(frEqual(elements[0], frFromInt(5)), "first element is coset shift")
    // All elements should be distinct
    var allDistinct = true
    for i in 0..<elements.count {
        for j in (i + 1)..<elements.count {
            if frEqual(elements[i], elements[j]) { allDistinct = false }
        }
    }
    expect(allDistinct, "all domain elements are distinct")
}

private func testCompositionDomainEvaluation() {
    let domain = CompositionDomain(logN: 3, cosetShift: frFromInt(3), blowupFactor: 4)
    let evalElements = domain.evaluationDomainElements()
    expectEqual(evalElements.count, 32, "evaluation domain has 32 elements")
    expect(frEqual(evalElements[0], frFromInt(3)), "first eval element is coset shift")
}

private func testCompositionDomainBlowup() {
    let d1 = CompositionDomain(logN: 4, cosetShift: Fr.one, blowupFactor: 1)
    expectEqual(d1.logBlowup, 0, "blowup 1 has logBlowup 0")
    let d2 = CompositionDomain(logN: 4, cosetShift: Fr.one, blowupFactor: 8)
    expectEqual(d2.logBlowup, 3, "blowup 8 has logBlowup 3")
    expectEqual(d2.evaluationDomainSize, 128, "eval domain 128 with blowup 8")
}

// MARK: - Engine Construction Tests

private func testEngineFromConstraints() {
    let constraints = compileFibConstraints()
    let engine = GPUAIRCompositionEngine(
        numMainColumns: 2,
        logTraceLength: logN,
        transitionConstraints: constraints)
    expectEqual(engine.numMainColumns, 2, "2 main columns")
    expectEqual(engine.logTraceLength, logN, "correct log trace length")
    expectEqual(engine.transitionConstraints.count, 2, "2 transition constraints")
    expectEqual(engine.numAuxColumns, 0, "no auxiliary columns")
}

private func testEngineFromAIR() {
    let air = try! GPUAIRConstraintCompiler.compile(
        numColumns: 2, logTraceLength: logN,
        transitions: [
            FrAIRExprBuilder.next(0) - FrAIRExprBuilder.col(1),
            FrAIRExprBuilder.next(1) - (FrAIRExprBuilder.col(0) + FrAIRExprBuilder.col(1))
        ],
        boundaries: [(column: 0, row: 0, value: Fr.one)],
        traceGenerator: { makeFibTrace() })
    let engine = GPUAIRCompositionEngine.fromAIR(air)
    expectEqual(engine.numMainColumns, 2, "fromAIR: 2 columns")
    expectEqual(engine.transitionConstraints.count, 2, "fromAIR: 2 constraints")
    expectEqual(engine.boundaryConstraints.count, 1, "fromAIR: 1 boundary")
}

private func testEngineDimensions() {
    let engine = makeFibEngine()
    expectEqual(engine.traceLength, traceLen, "trace length matches")
    expectEqual(engine.totalColumns, 2, "total columns = main + aux")
}

private func testEngineConstraintGroups() {
    // Fibonacci constraints are both degree 1
    let engine = makeFibEngine()
    let groups = engine.constraintGroups
    expectEqual(groups.count, 1, "all degree-1 constraints in one group")
    expectEqual(groups[0].degree, 1, "group degree is 1")
    expectEqual(groups[0].constraints.count, 2, "group has 2 constraints")
}

// MARK: - Single Constraint Tests

private func testSingleConstraintComposition() {
    let constraints = compileCounterConstraint()
    let engine = GPUAIRCompositionEngine(
        numMainColumns: 1, logTraceLength: logN,
        transitionConstraints: constraints)
    let trace = makeCounterTrace()
    let alpha = frFromInt(42)
    let domain = defaultDomain()
    let result = engine.compose(trace: trace, alpha: alpha, domain: domain)
    expectEqual(result.numConstraints, 1, "single constraint")
    expect(result.compositionTimeSeconds >= 0, "non-negative time")
}

private func testSingleConstraintChunks() {
    let constraints = compileCounterConstraint()
    let engine = GPUAIRCompositionEngine(
        numMainColumns: 1, logTraceLength: logN,
        transitionConstraints: constraints)
    let trace = makeCounterTrace()
    let alpha = frFromInt(7)
    let domain = defaultDomain()
    let result = engine.compose(trace: trace, alpha: alpha, domain: domain)
    expect(result.numChunks >= 1, "at least one chunk")
    for chunk in result.chunks {
        expect(chunk.evaluations.count > 0, "chunk has evaluations")
        expectEqual(chunk.degreeBound, traceLen, "chunk degree bound is trace length")
    }
}

private func testSingleConstraintValidTrace() {
    let constraints = compileCounterConstraint()
    let engine = GPUAIRCompositionEngine(
        numMainColumns: 1, logTraceLength: logN,
        transitionConstraints: constraints)
    let trace = makeCounterTrace()
    let evals = engine.evaluateConstraintsOnTrace(trace: trace)
    expectEqual(evals.count, 1, "one constraint evaluation array")
    // Valid counter trace: all evaluations should be 0 for rows 0..n-2
    var allZero = true
    for row in 0..<(traceLen - 1) {
        if !evals[0][row].isZero { allZero = false; break }
    }
    expect(allZero, "valid counter trace has all-zero constraint evals")
}

// MARK: - Multiple Constraint Tests

private func testMultipleConstraintComposition() {
    let engine = makeFibEngine()
    let trace = makeFibTrace()
    let alpha = frFromInt(13)
    let domain = defaultDomain()
    let result = engine.compose(trace: trace, alpha: alpha, domain: domain)
    expectEqual(result.numConstraints, 2, "2 constraints composed")
    expectEqual(result.evaluations.count, traceLen, "evaluations for all rows")
}

private func testMultipleConstraintAlphaPowers() {
    let engine = makeFibEngine()
    let trace = makeFibTrace()
    // With valid trace, composed evaluations should all be zero
    let evals = engine.evaluateConstraintsOnTrace(trace: trace)
    let alpha = frFromInt(100)
    let composed = engine.composeEvaluations(evals, alpha: alpha)
    var allZero = true
    for row in 0..<(traceLen - 1) {
        if !composed[row].isZero { allZero = false; break }
    }
    expect(allZero, "composed evaluations zero for valid Fibonacci trace")
}

private func testMultipleConstraintDegrees() {
    let engine = makeFibEngine()
    let analysis = engine.degreeAnalysis()
    expectEqual(analysis.transitionDegrees.count, 2, "2 degree entries")
    expectEqual(analysis.maxTransitionDegree, 1, "max degree is 1 for linear constraints")
}

// MARK: - Zerofier Tests

private func testZerofierOnDomain() {
    let engine = makeFibEngine()
    let omega = computeNthRootOfUnity(logN: logN)
    // Zerofier should vanish on the trace domain
    for i in 0..<traceLen {
        let point = frPow(omega, UInt64(i))
        let z = engine.evaluateZerofier(at: point)
        expect(z.isZero, "zerofier vanishes at omega^\(i)")
    }
}

private func testZerofierOffDomain() {
    let engine = makeFibEngine()
    // Off-domain point should give nonzero zerofier
    let offPoint = frFromInt(12345)
    let z = engine.evaluateZerofier(at: offPoint)
    expect(!z.isZero, "zerofier nonzero off-domain")
}

private func testZerofierOnCoset() {
    let engine = makeFibEngine()
    let domain = defaultDomain()
    let zerofierEvals = engine.evaluateZerofierOnDomain(domain)
    expectEqual(zerofierEvals.count, domain.evaluationDomainSize,
                "zerofier evaluated at all domain points")
    // Coset shift 7 is not on the trace domain, so zerofier should be nonzero
    var anyNonzero = false
    for z in zerofierEvals {
        if !z.isZero { anyNonzero = true; break }
    }
    expect(anyNonzero, "zerofier is nonzero on coset domain")
}

private func testBoundaryZerofier() {
    let engine = makeFibEngine()
    let omega = computeNthRootOfUnity(logN: logN)
    // Boundary zerofier at row 0: Z_B(x) = x - omega^0 = x - 1
    let zAtOne = engine.evaluateBoundaryZerofier(at: Fr.one, boundaryRow: 0)
    expect(zAtOne.isZero, "boundary zerofier vanishes at omega^0 = 1")
    // At row 3: Z_B(x) = x - omega^3
    let omega3 = frPow(omega, 3)
    let zAtOmega3 = engine.evaluateBoundaryZerofier(at: omega3, boundaryRow: 3)
    expect(zAtOmega3.isZero, "boundary zerofier vanishes at omega^3")
}

// MARK: - Chunk Splitting Tests

private func testChunkSplitSingle() {
    let engine = GPUAIRCompositionEngine(
        numMainColumns: 1, logTraceLength: logN,
        transitionConstraints: compileCounterConstraint())
    let evals = [Fr](repeating: frFromInt(1), count: traceLen)
    let domain = defaultDomain()
    let chunks = engine.splitIntoChunks(evals, domain: domain)
    expectEqual(chunks.count, 1, "degree-1 constraint gives 1 chunk")
    expectEqual(chunks[0].index, 0, "first chunk has index 0")
    expectEqual(chunks[0].evaluations.count, traceLen, "chunk has traceLen evaluations")
}

private func testChunkSplitMultiple() {
    // Create a quadratic constraint for multiple chunks
    let expr = FrAIRExprBuilder.col(0) * FrAIRExprBuilder.col(0)
    let constraint = try! GPUAIRConstraintCompiler.compileExpression(expr, numColumns: 1)
    let engine = GPUAIRCompositionEngine(
        numMainColumns: 1, logTraceLength: logN,
        transitionConstraints: [constraint],
        constraintDegrees: [2])
    let evals = [Fr](repeating: frFromInt(5), count: 2 * traceLen)
    let domain = defaultDomain()
    let chunks = engine.splitIntoChunks(evals, domain: domain)
    expectEqual(chunks.count, 2, "degree-2 constraint gives 2 chunks")
    expectEqual(chunks[1].index, 1, "second chunk has index 1")
}

private func testChunkDegreeBounds() {
    let engine = makeFibEngine()
    let evals = [Fr](repeating: Fr.zero, count: traceLen)
    let domain = defaultDomain()
    let chunks = engine.splitIntoChunks(evals, domain: domain)
    for chunk in chunks {
        expectEqual(chunk.degreeBound, traceLen, "chunk degree bound = trace length")
    }
}

// MARK: - Multi-Trace Tests

private func testAuxiliaryConstraintEval() {
    // aux constraint: aux_col[0]' = main_col[0] + aux_col[0]
    let expr = FrAIRExprBuilder.next(1) - (FrAIRExprBuilder.col(0) + FrAIRExprBuilder.col(1))
    let compiled = try! GPUAIRConstraintCompiler.compileExpression(expr, numColumns: 2)
    let auxC = AuxiliaryConstraint(
        constraint: compiled, numMainColumns: 1, numAuxColumns: 1,
        label: "accumulator")
    // main trace: [0, 1, 2, ...], aux trace: running sum [0, 0, 1, 3, ...]
    let mainCurrent: [Fr] = [frFromInt(2)]
    let mainNext: [Fr] = [frFromInt(3)]
    let auxCurrent: [Fr] = [frFromInt(5)]
    let auxNext: [Fr] = [frFromInt(7)] // 2 + 5 = 7
    let result = auxC.evaluate(
        mainCurrent: mainCurrent, mainNext: mainNext,
        auxCurrent: auxCurrent, auxNext: auxNext)
    expect(result.isZero, "auxiliary constraint satisfied when next_aux = main + aux")
}

private func testMultiTraceComposition() {
    let mainConstraint = compileCounterConstraint()
    // Auxiliary constraint: aux' = main + aux (compiled over 2 total columns)
    let auxExpr = FrAIRExprBuilder.next(1) - (FrAIRExprBuilder.col(0) + FrAIRExprBuilder.col(1))
    let auxCompiled = try! GPUAIRConstraintCompiler.compileExpression(auxExpr, numColumns: 2)
    let auxC = AuxiliaryConstraint(
        constraint: auxCompiled, numMainColumns: 1, numAuxColumns: 1,
        label: "running_sum")
    let engine = GPUAIRCompositionEngine(
        numMainColumns: 1, logTraceLength: logN,
        transitionConstraints: mainConstraint,
        auxiliaryConstraints: [auxC],
        numAuxColumns: 1)
    expectEqual(engine.totalColumns, 2, "1 main + 1 aux = 2 total")
    expectEqual(engine.auxiliaryConstraints.count, 1, "1 auxiliary constraint")
}

private func testMultiTraceResult() {
    let mainConstraint = compileCounterConstraint()
    let auxExpr = FrAIRExprBuilder.next(1) - (FrAIRExprBuilder.col(0) + FrAIRExprBuilder.col(1))
    let auxCompiled = try! GPUAIRConstraintCompiler.compileExpression(auxExpr, numColumns: 2)
    let auxC = AuxiliaryConstraint(
        constraint: auxCompiled, numMainColumns: 1, numAuxColumns: 1)
    let engine = GPUAIRCompositionEngine(
        numMainColumns: 1, logTraceLength: logN,
        transitionConstraints: mainConstraint,
        auxiliaryConstraints: [auxC],
        numAuxColumns: 1)

    let mainTrace = makeCounterTrace()
    // aux trace: running sum of counter: 0, 0, 1, 3, 6, 10, ...
    var auxCol = [Fr](repeating: Fr.zero, count: traceLen)
    for i in 1..<traceLen {
        auxCol[i] = frAdd(auxCol[i - 1], mainTrace[0][i - 1])
    }
    let auxTrace = [auxCol]

    let alpha = frFromInt(17)
    let domain = defaultDomain()
    let result = engine.composeMultiTrace(
        mainTrace: mainTrace, auxTrace: auxTrace, alpha: alpha, domain: domain)
    expectEqual(result.numConstraints, 2, "1 main + 1 aux = 2 total constraints")
    expect(result.evaluations.count > 0, "has evaluations")
}

// MARK: - Deep Composition Tests

private func testDeepCompositionCoeffs() {
    let coeffs = DeepCompositionCoefficients.random(
        numTraceColumns: 3, numChunks: 2, seed: frFromInt(42))
    expectEqual(coeffs.traceCoeffs.count, 3, "3 trace coefficients")
    expectEqual(coeffs.traceNextCoeffs.count, 3, "3 trace-next coefficients")
    expectEqual(coeffs.compositionCoeffs.count, 2, "2 composition coefficients")
    // All coefficients should be nonzero (with overwhelming probability)
    var anyZeroTrace = false
    for c in coeffs.traceCoeffs { if c.isZero { anyZeroTrace = true } }
    expect(!anyZeroTrace, "trace coefficients are nonzero")
}

private func testDeepCompositionBasic() {
    let engine = makeFibEngine()
    let trace = makeFibTrace()
    let n = 4 // small domain for test
    let omega = computeNthRootOfUnity(logN: logN)
    var domainPoints = [Fr]()
    let shift = frFromInt(7)
    for i in 0..<n {
        domainPoints.append(frMul(shift, frPow(omega, UInt64(i))))
    }
    // Evaluate trace at domain points (use trace values directly for simplicity)
    let traceEvals = trace.map { Array($0.prefix(n)) }
    let chunkEvals = [[Fr](repeating: frFromInt(1), count: n)]
    let traceAtZ = [frFromInt(1), frFromInt(2)]
    let traceAtZOmega = [frFromInt(3), frFromInt(4)]
    let chunksAtZ = [frFromInt(5)]
    let z = frFromInt(999)
    let coeffs = DeepCompositionCoefficients(
        traceCoeffs: [frFromInt(1), frFromInt(1)],
        traceNextCoeffs: [frFromInt(1), frFromInt(1)],
        compositionCoeffs: [frFromInt(1)])
    let deepResult = engine.deepCompose(
        traceEvals: traceEvals, chunkEvals: chunkEvals,
        traceAtZ: traceAtZ, traceAtZOmega: traceAtZOmega,
        chunksAtZ: chunksAtZ, z: z, domainPoints: domainPoints,
        coeffs: coeffs)
    expectEqual(deepResult.count, n, "deep composition has n evaluations")
}

// MARK: - Parallel Composition Tests

private func testParallelComposeMatchesSequential() {
    let engine = makeFibEngine()
    let trace = makeFibTrace()
    let alpha = frFromInt(31)
    let domain = defaultDomain()
    let seqResult = engine.compose(trace: trace, alpha: alpha, domain: domain)
    let parResult = engine.parallelCompose(
        trace: trace, alpha: alpha, domain: domain, blockSize: 4)
    // Both should produce the same evaluations
    expectEqual(seqResult.evaluations.count, parResult.evaluations.count,
                "same number of evaluations")
    var match = true
    for i in 0..<min(seqResult.evaluations.count, parResult.evaluations.count) {
        if !frEqual(seqResult.evaluations[i], parResult.evaluations[i]) {
            match = false; break
        }
    }
    expect(match, "parallel and sequential produce identical evaluations")
}

private func testParallelComposeBlockSize() {
    let engine = makeFibEngine()
    let trace = makeFibTrace()
    let alpha = frFromInt(7)
    let domain = defaultDomain()
    // Try different block sizes
    let r1 = engine.parallelCompose(trace: trace, alpha: alpha, domain: domain, blockSize: 2)
    let r2 = engine.parallelCompose(trace: trace, alpha: alpha, domain: domain, blockSize: 8)
    var match = true
    for i in 0..<min(r1.evaluations.count, r2.evaluations.count) {
        if !frEqual(r1.evaluations[i], r2.evaluations[i]) {
            match = false; break
        }
    }
    expect(match, "different block sizes produce identical results")
}

// MARK: - Boundary Quotient Tests

private func testBoundaryQuotientSingle() {
    let engine = makeFibEngine()
    let trace = makeFibTrace()
    let domain = CompositionDomain(logN: logN, cosetShift: frFromInt(7), blowupFactor: 1)
    let quotients = engine.evaluateBoundaryQuotients(trace: trace, domain: domain)
    expectEqual(quotients.count, 2, "2 boundary quotients (2 boundary constraints)")
    expectEqual(quotients[0].count, traceLen, "quotient has traceLen evaluations")
}

private func testBoundaryQuotientMultiple() {
    let constraints = compileFibConstraints()
    let engine = GPUAIRCompositionEngine(
        numMainColumns: 2, logTraceLength: logN,
        transitionConstraints: constraints,
        boundaryConstraints: [
            (column: 0, row: 0, value: Fr.one),
            (column: 1, row: 0, value: Fr.one),
            (column: 0, row: 1, value: Fr.one)
        ])
    let trace = makeFibTrace()
    let domain = CompositionDomain(logN: logN, cosetShift: frFromInt(11), blowupFactor: 1)
    let quotients = engine.evaluateBoundaryQuotients(trace: trace, domain: domain)
    expectEqual(quotients.count, 3, "3 boundary quotients")
}

// MARK: - Degree Analysis Tests

private func testAIRCompDegreeAnalysisLinear() {
    let engine = makeFibEngine()
    let analysis = engine.degreeAnalysis()
    expectEqual(analysis.maxTransitionDegree, 1, "fibonacci has degree 1")
    expectEqual(analysis.numQuotientChunks, 1, "1 quotient chunk for degree 1")
}

private func testAIRCompDegreeAnalysisQuadratic() {
    let expr = FrAIRExprBuilder.col(0) * FrAIRExprBuilder.col(0)
    let constraint = try! GPUAIRConstraintCompiler.compileExpression(expr, numColumns: 1)
    let engine = GPUAIRCompositionEngine(
        numMainColumns: 1, logTraceLength: logN,
        transitionConstraints: [constraint])
    let analysis = engine.degreeAnalysis()
    expectEqual(analysis.maxTransitionDegree, 2, "quadratic has degree 2")
    expectEqual(analysis.numQuotientChunks, 2, "2 quotient chunks for degree 2")
}

private func testDegreeBoundComputation() {
    let engine = makeFibEngine()
    let bound = engine.compositionDegreeBound()
    expectEqual(bound, traceLen, "degree bound = maxDeg * traceLen = 1 * 16")
    let numChunks = engine.numQuotientChunks()
    expectEqual(numChunks, 1, "1 chunk for degree-1 constraints")
}

// MARK: - Verification Tests

private func testVerifyValidTrace() {
    let engine = makeFibEngine()
    let trace = makeFibTrace()
    let alpha = frFromInt(42)
    let domain = defaultDomain()
    let result = engine.compose(trace: trace, alpha: alpha, domain: domain)
    let ok = engine.verifyComposition(trace: trace, result: result, numSamples: 3)
    expect(ok, "verification passes for valid Fibonacci trace")
}

private func testVerifyInvalidTrace() {
    let engine = makeFibEngine()
    // Create an invalid trace by corrupting one value
    var trace = makeFibTrace()
    trace[0][5] = frFromInt(9999)
    let evals = engine.evaluateConstraintsOnTrace(trace: trace)
    // Should have nonzero evaluations near the corrupted row
    var hasNonzero = false
    for ci in 0..<evals.count {
        for row in 0..<(traceLen - 1) {
            if !evals[ci][row].isZero { hasNonzero = true; break }
        }
        if hasNonzero { break }
    }
    expect(hasNonzero, "invalid trace produces nonzero constraint evaluations")
}

// MARK: - Fibonacci AIR Integration Tests

private func testFibonacciComposition() {
    let engine = makeFibEngine()
    let trace = makeFibTrace()
    let alpha = frFromInt(73)
    let domain = defaultDomain()
    let result = engine.compose(trace: trace, alpha: alpha, domain: domain)
    // For a valid trace, the composed evaluations on the trace domain should be zero
    var allZero = true
    for row in 0..<(traceLen - 1) {
        if !result.evaluations[row].isZero { allZero = false; break }
    }
    expect(allZero, "Fibonacci composition yields all-zero evals for valid trace")
}

private func testFibonacciChunks() {
    let engine = makeFibEngine()
    let trace = makeFibTrace()
    let alpha = frFromInt(7)
    let domain = defaultDomain()
    let result = engine.compose(trace: trace, alpha: alpha, domain: domain)
    // Fibonacci constraints are degree 1 => 1 chunk
    expectEqual(result.numChunks, 1, "fibonacci AIR produces 1 composition chunk")
    let chunk = result.chunks[0]
    expectEqual(chunk.index, 0, "chunk index is 0")
    // All chunk evaluations should be zero for valid trace
    var chunkAllZero = true
    for val in chunk.evaluations {
        if !val.isZero { chunkAllZero = false; break }
    }
    expect(chunkAllZero, "chunk evaluations all zero for valid Fibonacci trace")
}

// MARK: - Edge Case Tests

private func testSingleRowTrace() {
    // logN=1 gives trace of 2 rows
    let c = try! GPUAIRConstraintCompiler.compileExpression(
        FrAIRExprBuilder.next(0) - FrAIRExprBuilder.col(0),
        numColumns: 1)
    let engine = GPUAIRCompositionEngine(
        numMainColumns: 1, logTraceLength: 1,
        transitionConstraints: [c])
    let trace: [[Fr]] = [[Fr.one, Fr.one]]
    let domain = CompositionDomain(logN: 1, cosetShift: frFromInt(3), blowupFactor: 2)
    let result = engine.compose(trace: trace, alpha: frFromInt(1), domain: domain)
    // col' = col should be satisfied for [1, 1]
    expect(result.evaluations[0].isZero, "identity constraint satisfied")
}

private func testLargeAlpha() {
    let engine = makeFibEngine()
    let trace = makeFibTrace()
    // Very large alpha
    let largeAlpha = frPow(frFromInt(2), 200)
    let domain = defaultDomain()
    let result = engine.compose(trace: trace, alpha: largeAlpha, domain: domain)
    // Valid trace should still give zero evaluations regardless of alpha
    var allZero = true
    for row in 0..<(traceLen - 1) {
        if !result.evaluations[row].isZero { allZero = false; break }
    }
    expect(allZero, "large alpha still gives zero evals for valid trace")
}

private func testZeroAlpha() {
    let engine = makeFibEngine()
    let trace = makeFibTrace()
    let domain = defaultDomain()
    let result = engine.compose(trace: trace, alpha: Fr.zero, domain: domain)
    // With alpha=0, only the first constraint contributes (alpha^0 = 1, alpha^1 = 0)
    expectEqual(result.numConstraints, 2, "still reports 2 constraints")
    // Valid trace => still zero
    var allZero = true
    for row in 0..<(traceLen - 1) {
        if !result.evaluations[row].isZero { allZero = false; break }
    }
    expect(allZero, "zero alpha still gives zero evals for valid trace")
}

private func testComposeAll() {
    let engine = makeFibEngine()
    let trace = makeFibTrace()
    let alpha = frFromInt(53)
    let domain = CompositionDomain(logN: logN, cosetShift: frFromInt(7), blowupFactor: 1)
    let result = engine.composeAll(trace: trace, alpha: alpha, domain: domain)
    // Should include both transition and boundary constraints
    expectEqual(result.numConstraints, 4, "2 transition + 2 boundary = 4 total")
    expect(result.evaluations.count > 0, "has evaluations")
}
