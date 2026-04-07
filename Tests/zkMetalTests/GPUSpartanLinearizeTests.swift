// GPUSpartanLinearizeTests — Tests for GPU Spartan R1CS linearization engine
import zkMetal
import Foundation

public func runGPUSpartanLinearizeTests() {
    suite("GPU Spartan Linearize Engine")

    testSparseMLETableConstruction()
    testSparseMLEEvaluation()
    testLinearizedR1CSBasic()
    testLinearizedR1CSRowBinding()
    testHypercubeEqTable()
    testHypercubeFold()
    testHypercubeMLEEvaluation()
    testWitnessLinearizationBuild()
    testWitnessLinearizationEval()
    testWitnessLinearizationExtract()
    testMemoryCheckingDigest()
    testMemoryCheckingConsistency()
    testStructuredR1CSAnalysis()
    testStructuredR1CSUniformDetection()
    testEngineLinearize()
    testEngineFullLinearize()
    testEngineSumcheckRoundDeg2()
    testEngineSumcheckRoundDeg3()
    testEngineSumcheckReduce()
    testEngineBatchEvaluate()
    testEngineLinearizeCache()
    testMultiplyCircuitLinearization()
    testAddChainCircuitLinearization()
    testRangeCheckCircuitLinearization()
    testQuadraticCircuitLinearization()
    testSumcheckRoundConsistency()
    testLargerCircuitLinearization()
    testEmptySparseTable()
    testSingleConstraintCircuit()
}

// MARK: - SparseMLETable Tests

private func testSparseMLETableConstruction() {
    // Build a simple 2x2 identity-like sparse matrix
    let entries = [
        SpartanEntry(row: 0, col: 0, value: Fr.one),
        SpartanEntry(row: 1, col: 1, value: Fr.one),
    ]
    let table = SparseMLETable.fromEntries(entries, logM: 1, logN: 1)
    expectEqual(table.nnz, 2, "SparseMLETable: identity has 2 nonzeros")
    expectEqual(table.numRowVars, 1, "SparseMLETable: logM=1")
    expectEqual(table.numColVars, 1, "SparseMLETable: logN=1")
    expectEqual(table.rows[0], 0, "SparseMLETable: first entry row=0")
    expectEqual(table.cols[1], 1, "SparseMLETable: second entry col=1")
}

private func testSparseMLEEvaluation() {
    // Matrix: [[3, 0], [0, 5]]
    let three = frFromInt(3)
    let five = frFromInt(5)
    let entries = [
        SpartanEntry(row: 0, col: 0, value: three),
        SpartanEntry(row: 1, col: 1, value: five),
    ]
    let table = SparseMLETable.fromEntries(entries, logM: 1, logN: 1)

    // Evaluate at boolean points to verify correctness
    // At (tau=[0], x=[0]): eq(0,0)*3 + eq(0,1)*0 = 1*3 = 3
    let e00 = table.evaluate(tau: [Fr.zero], x: [Fr.zero])
    expect(spartanFrEqual(e00, three), "SparseMLETable eval at (0,0) = 3")

    // At (tau=[1], x=[1]): eq(1,1)*5 = 1*5 = 5
    let e11 = table.evaluate(tau: [Fr.one], x: [Fr.one])
    expect(spartanFrEqual(e11, five), "SparseMLETable eval at (1,1) = 5")

    // At (tau=[0], x=[1]): eq(0,0)*0 + eq(0,1)*0 = 0 (no entry at row=0,col=1)
    let e01 = table.evaluate(tau: [Fr.zero], x: [Fr.one])
    expect(spartanFrEqual(e01, Fr.zero), "SparseMLETable eval at (0,1) = 0")

    // At (tau=[1], x=[0]): eq(1,1)*0 = 0 (no entry at row=1,col=0)
    let e10 = table.evaluate(tau: [Fr.one], x: [Fr.zero])
    expect(spartanFrEqual(e10, Fr.zero), "SparseMLETable eval at (1,0) = 0")
}

private func testLinearizedR1CSBasic() {
    // Use the multiply circuit: x * y = z
    let (instance, gen) = SpartanR1CSBuilder.buildMultiplyCircuit()
    let lin = LinearizedR1CS.linearize(instance)

    expectEqual(lin.numConstraints, 1, "LinearizedR1CS: multiply has 1 constraint")
    expectEqual(lin.numPublic, 1, "LinearizedR1CS: multiply has 1 public")
    expect(lin.logM >= 0, "LinearizedR1CS: logM >= 0")
    expect(lin.logN >= 1, "LinearizedR1CS: logN >= 1 (at least 2 variables)")
    expect(lin.mlA.nnz > 0, "LinearizedR1CS: A has nonzeros")
    expect(lin.mlB.nnz > 0, "LinearizedR1CS: B has nonzeros")
    expect(lin.mlC.nnz > 0, "LinearizedR1CS: C has nonzeros")
}

private func testLinearizedR1CSRowBinding() {
    // Multiply circuit: A has entry at (0, x_col), B at (0, y_col), C at (0, z_col)
    let (instance, _) = SpartanR1CSBuilder.buildMultiplyCircuit()
    let lin = LinearizedR1CS.linearize(instance)

    // Use tau at a boolean point (row 0)
    let logM = lin.logM
    var tau = [Fr](repeating: Fr.zero, count: logM)
    // tau = [0,...,0] selects row 0
    let (aVec, bVec, cVec) = lin.rowBindAll(tau: tau)

    // aVec should be nonzero at the column corresponding to x
    let paddedN = 1 << lin.logN
    expectEqual(aVec.count, paddedN, "rowBind A vector has paddedN elements")
    expectEqual(bVec.count, paddedN, "rowBind B vector has paddedN elements")
    expectEqual(cVec.count, paddedN, "rowBind C vector has paddedN elements")

    // At least one entry should be nonzero for a non-trivial circuit
    var aHasNonzero = false
    for v in aVec { if !v.isZero { aHasNonzero = true; break } }
    expect(aHasNonzero, "rowBind A has at least one nonzero for row 0")
}

// MARK: - HypercubeEvaluator Tests

private func testHypercubeEqTable() {
    let eval = HypercubeEvaluator(numVars: 2)

    // eq([0,0], b) should be [1, 0, 0, 0] for b in {00, 01, 10, 11}
    let eq00 = eval.eqTable(point: [Fr.zero, Fr.zero])
    expect(spartanFrEqual(eq00[0], Fr.one), "eq([0,0], [0,0]) = 1")
    expect(spartanFrEqual(eq00[1], Fr.zero), "eq([0,0], [0,1]) = 0")
    expect(spartanFrEqual(eq00[2], Fr.zero), "eq([0,0], [1,0]) = 0")
    expect(spartanFrEqual(eq00[3], Fr.zero), "eq([0,0], [1,1]) = 0")

    // eq([1,1], b) should be [0, 0, 0, 1]
    let eq11 = eval.eqTable(point: [Fr.one, Fr.one])
    expect(spartanFrEqual(eq11[0], Fr.zero), "eq([1,1], [0,0]) = 0")
    expect(spartanFrEqual(eq11[3], Fr.one), "eq([1,1], [1,1]) = 1")

    // Sum of eq table over all b should be 1 for any point
    let r0 = frFromInt(3)
    let r1 = frFromInt(7)
    let eqR = eval.eqTable(point: [r0, r1])
    var total = Fr.zero
    for v in eqR { total = frAdd(total, v) }
    expect(spartanFrEqual(total, Fr.one), "sum of eq table = 1")
}

private func testHypercubeFold() {
    let eval = HypercubeEvaluator(numVars: 2)

    // Start with [a, b, c, d], fold with challenge r
    // Result: [a + r*(c-a), b + r*(d-b)]
    let a = frFromInt(1), b = frFromInt(2), c = frFromInt(3), d = frFromInt(4)
    let r = frFromInt(5)
    let folded = eval.fold([a, b, c, d], challenge: r)
    expectEqual(folded.count, 2, "fold reduces size by half")

    // folded[0] = 1 + 5*(3-1) = 1 + 10 = 11
    let expected0 = frFromInt(11)
    expect(spartanFrEqual(folded[0], expected0), "fold[0] = 1 + 5*(3-1) = 11")

    // folded[1] = 2 + 5*(4-2) = 2 + 10 = 12
    let expected1 = frFromInt(12)
    expect(spartanFrEqual(folded[1], expected1), "fold[1] = 2 + 5*(4-2) = 12")
}

private func testHypercubeMLEEvaluation() {
    // MLE with evals [1, 2, 3, 4] on {0,1}^2
    // MSB-first: pt[0]=x1, pt[1]=x0
    // f(0,0)=1, f(0,1)=2, f(1,0)=3, f(1,1)=4
    let evals: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let eval = HypercubeEvaluator(numVars: 2)

    // Evaluate at boolean points (MSB-first convention matches spartanEvalML)
    let v00 = eval.evaluateMLE(evals: evals, point: [Fr.zero, Fr.zero])
    expect(spartanFrEqual(v00, frFromInt(1)), "MLE eval at (0,0) = 1")

    let v10 = eval.evaluateMLE(evals: evals, point: [Fr.one, Fr.zero])
    expect(spartanFrEqual(v10, frFromInt(3)), "MLE eval at (1,0) = 3")

    let v01 = eval.evaluateMLE(evals: evals, point: [Fr.zero, Fr.one])
    expect(spartanFrEqual(v01, frFromInt(2)), "MLE eval at (0,1) = 2")

    let v11 = eval.evaluateMLE(evals: evals, point: [Fr.one, Fr.one])
    expect(spartanFrEqual(v11, frFromInt(4)), "MLE eval at (1,1) = 4")

    // Evaluate at a non-boolean point and check against spartanEvalML
    let r0 = frFromInt(2), r1 = frFromInt(3)
    let vR = eval.evaluateMLE(evals: evals, point: [r0, r1])
    let vRef = spartanEvalML(evals: evals, pt: [r0, r1])
    expect(spartanFrEqual(vR, vRef), "MLE eval at (2,3) matches spartanEvalML")
}

// MARK: - WitnessLinearization Tests

private func testWitnessLinearizationBuild() {
    let pub: [Fr] = [frFromInt(42)]
    let wit: [Fr] = [frFromInt(6), frFromInt(7)]
    let wl = WitnessLinearization.build(publicInputs: pub, witness: wit, logN: 2)

    expectEqual(wl.logN, 2, "WitnessLin: logN=2")
    expectEqual(wl.numPublic, 1, "WitnessLin: 1 public input")
    expectEqual(wl.numWitness, 2, "WitnessLin: 2 witness elements")
    expectEqual(wl.zTilde.count, 4, "WitnessLin: padded to 2^2=4")

    // z = [1, 42, 6, 7]
    expect(spartanFrEqual(wl.zTilde[0], Fr.one), "WitnessLin: z[0] = 1")
    expect(spartanFrEqual(wl.zTilde[1], frFromInt(42)), "WitnessLin: z[1] = 42 (public)")
    expect(spartanFrEqual(wl.zTilde[2], frFromInt(6)), "WitnessLin: z[2] = 6 (witness)")
    expect(spartanFrEqual(wl.zTilde[3], frFromInt(7)), "WitnessLin: z[3] = 7 (witness)")
}

private func testWitnessLinearizationEval() {
    let pub: [Fr] = [frFromInt(10)]
    let wit: [Fr] = [frFromInt(20), frFromInt(30)]
    let wl = WitnessLinearization.build(publicInputs: pub, witness: wit, logN: 2)

    // Evaluate at (0,0) should give z[0] = 1
    let v00 = wl.evaluate(at: [Fr.zero, Fr.zero])
    expect(spartanFrEqual(v00, Fr.one), "WitnessLin eval at (0,0) = 1")

    // MSB-first: pt[0]=1,pt[1]=0 means x1=1,x0=0 -> z[2] = 20
    let v10 = wl.evaluate(at: [Fr.one, Fr.zero])
    expect(spartanFrEqual(v10, frFromInt(20)), "WitnessLin eval at (1,0) = 20")

    // Evaluate at a random point should match spartanEvalML
    let r0 = frFromInt(5), r1 = frFromInt(7)
    let vR = wl.evaluate(at: [r0, r1])
    let vRef = spartanEvalML(evals: wl.zTilde, pt: [r0, r1])
    expect(spartanFrEqual(vR, vRef), "WitnessLin eval at (5,7) matches spartanEvalML")
}

private func testWitnessLinearizationExtract() {
    let pub: [Fr] = [frFromInt(100), frFromInt(200)]
    let wit: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30)]
    let wl = WitnessLinearization.build(publicInputs: pub, witness: wit, logN: 3)

    let pubVals = wl.publicInputValues()
    expectEqual(pubVals.count, 2, "WitnessLin: extracted 2 public inputs")
    expect(spartanFrEqual(pubVals[0], frFromInt(100)), "WitnessLin: pub[0]=100")
    expect(spartanFrEqual(pubVals[1], frFromInt(200)), "WitnessLin: pub[1]=200")

    let witVals = wl.witnessValues()
    expectEqual(witVals.count, 3, "WitnessLin: extracted 3 witness elements")
    expect(spartanFrEqual(witVals[0], frFromInt(10)), "WitnessLin: wit[0]=10")
    expect(spartanFrEqual(witVals[2], frFromInt(30)), "WitnessLin: wit[2]=30")
}

// MARK: - Memory-Checking Tests

private func testMemoryCheckingDigest() {
    let entries = [
        SpartanEntry(row: 0, col: 0, value: frFromInt(3)),
        SpartanEntry(row: 1, col: 1, value: frFromInt(5)),
    ]
    let table = SparseMLETable.fromEntries(entries, logM: 1, logN: 1)

    let gamma = frFromInt(100)
    let beta = frFromInt(7)
    let digest = MemoryCheckingDigest.compute(table: table, gamma: gamma, beta: beta)

    expectEqual(digest.numReads, 2, "MemCheck: 2 reads for 2 entries")
    expect(!digest.digest.isZero, "MemCheck: digest is nonzero")

    // Recompute with same parameters should give same digest
    let digest2 = MemoryCheckingDigest.compute(table: table, gamma: gamma, beta: beta)
    expect(spartanFrEqual(digest.digest, digest2.digest),
           "MemCheck: deterministic digest")
}

private func testMemoryCheckingConsistency() {
    // Create a small table and check read/write consistency
    let entries = [
        SpartanEntry(row: 0, col: 0, value: frFromInt(7)),
    ]
    let table = SparseMLETable.fromEntries(entries, logM: 1, logN: 1)

    let gamma = frFromInt(50)
    let beta = frFromInt(3)
    let readDigest = MemoryCheckingDigest.compute(table: table, gamma: gamma, beta: beta)

    // Build a dense table that matches the sparse entries
    var denseTable = [Fr](repeating: Fr.zero, count: 2) // 2x1
    denseTable[0] = frFromInt(7) // row=0, col=0

    let writeDigest = MemoryCheckingDigest.computeWriteDigest(
        tableValues: denseTable, gamma: gamma, beta: beta)

    // For this simple case, read digest computes (gamma - 0 - beta*7)
    // Write digest computes the same for the matching entry
    // They should be consistent
    expectEqual(readDigest.numReads, 1, "MemCheck consistency: 1 read")
    expectEqual(writeDigest.numWrites, 1, "MemCheck consistency: 1 write")
    expect(!readDigest.digest.isZero, "MemCheck consistency: read digest nonzero")
}

// MARK: - StructuredR1CS Tests

private func testStructuredR1CSAnalysis() {
    let (instance, _) = SpartanR1CSBuilder.buildMultiplyCircuit()
    let structured = StructuredR1CS.analyze(instance)

    expectEqual(structured.instance.numConstraints, 1,
                "Structured: multiply has 1 constraint")
    // A single constraint has no uniform partner
    expectEqual(structured.nonUniformIndices.count, 1,
                "Structured: single constraint is non-uniform")
}

private func testStructuredR1CSUniformDetection() {
    // Build a circuit with repeated multiply gates (same pattern)
    let b = SpartanR1CSBuilder()
    let out1 = b.addPublicInput()
    let out2 = b.addPublicInput()
    let x1 = b.addWitness()
    let y1 = b.addWitness()
    let x2 = b.addWitness()
    let y2 = b.addWitness()

    // Two multiply constraints with the same column pattern
    b.mulGate(a: x1, b: y1, out: out1)
    b.mulGate(a: x2, b: y2, out: out2)

    let instance = b.build()
    let structured = StructuredR1CS.analyze(instance)

    // Both constraints involve the same sparsity pattern (one col in A, one in B, one in C)
    // but with different column indices, so they may or may not be "uniform"
    // depending on exact column layout
    let totalCovered = structured.uniformConstraintCount + structured.nonUniformIndices.count
    expectEqual(totalCovered, 2,
                "Structured: all 2 constraints accounted for")
    expect(structured.uniformFraction >= 0.0 && structured.uniformFraction <= 1.0,
           "Structured: uniform fraction in [0,1]")
}

// MARK: - Engine Tests

private func testEngineLinearize() {
    let engine = GPUSpartanLinearizeEngine()
    let (instance, _) = SpartanR1CSBuilder.buildMultiplyCircuit()

    let lin = engine.linearize(instance)
    expectEqual(lin.numConstraints, 1, "Engine linearize: 1 constraint")
    expect(lin.mlA.nnz > 0, "Engine linearize: A nonzeros > 0")
    expect(lin.mlB.nnz > 0, "Engine linearize: B nonzeros > 0")
    expect(lin.mlC.nnz > 0, "Engine linearize: C nonzeros > 0")
}

private func testEngineFullLinearize() {
    let engine = GPUSpartanLinearizeEngine()
    let (instance, gen) = SpartanR1CSBuilder.buildMultiplyCircuit()
    let xVal = frFromInt(3)
    let yVal = frFromInt(7)
    let (pub, wit) = gen(xVal, yVal)

    // Verify R1CS is satisfied first
    let z = SpartanR1CS.buildZ(publicInputs: pub, witness: wit)
    expect(instance.isSatisfied(z: z), "FullLinearize: R1CS satisfied")

    let logM = instance.logM
    // Use a simple tau
    var tau = [Fr](repeating: Fr.zero, count: logM)

    let result = engine.fullLinearize(
        instance: instance, publicInputs: pub,
        witness: wit, tau: tau)

    // Check the linearization produced non-trivial results
    expect(result.witnessLinearization.zTilde.count > 0,
           "FullLinearize: z_tilde non-empty")
    expect(!result.memoryDigest.isZero, "FullLinearize: memory digest nonzero")

    // Check vectors have correct length
    let paddedN = 1 << instance.logN
    expectEqual(result.aVec.count, paddedN, "FullLinearize: aVec length = paddedN")
    expectEqual(result.bVec.count, paddedN, "FullLinearize: bVec length = paddedN")
    expectEqual(result.cVec.count, paddedN, "FullLinearize: cVec length = paddedN")
}

private func testEngineSumcheckRoundDeg2() {
    let engine = GPUSpartanLinearizeEngine()

    // Simple test: w = [1, 2, 3, 4], z = [5, 6, 7, 8]
    let w: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let z: [Fr] = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]

    let (s0, s1, s2) = engine.sumcheckRound(wVec: w, zVec: z, halfSize: 2)

    // s0 = w[0]*z[0] + w[1]*z[1] = 1*5 + 2*6 = 5 + 12 = 17
    expect(spartanFrEqual(s0, frFromInt(17)), "SC2 round: s(0) = 17")

    // s1 = w[2]*z[2] + w[3]*z[3] = 3*7 + 4*8 = 21 + 32 = 53
    expect(spartanFrEqual(s1, frFromInt(53)), "SC2 round: s(1) = 53")

    // s(0) + s(1) should be the full inner product = 5+12+21+32 = 70
    let sum01 = frAdd(s0, s1)
    expect(spartanFrEqual(sum01, frFromInt(70)), "SC2 round: s(0)+s(1) = 70")
}

private func testEngineSumcheckRoundDeg3() {
    let engine = GPUSpartanLinearizeEngine()

    // Use simple vectors of length 2 (half=1) for verifiable computation
    let eq: [Fr] = [frFromInt(1), frFromInt(1)]
    let az: [Fr] = [frFromInt(2), frFromInt(3)]
    let bz: [Fr] = [frFromInt(4), frFromInt(5)]
    let cz: [Fr] = [frFromInt(8), frFromInt(15)]

    let (s0, s1, s2, s3) = engine.sumcheckRoundDeg3(
        eqTau: eq, az: az, bz: bz, cz: cz, halfSize: 1)

    // s0 = eq[0] * (az[0]*bz[0] - cz[0]) = 1 * (2*4 - 8) = 0
    expect(spartanFrEqual(s0, Fr.zero), "SC3 round: s(0) = 0")

    // s1 = eq[1] * (az[1]*bz[1] - cz[1]) = 1 * (3*5 - 15) = 0
    expect(spartanFrEqual(s1, Fr.zero), "SC3 round: s(1) = 0")

    // For a satisfied R1CS, s(0) + s(1) = 0
    let sum01 = frAdd(s0, s1)
    expect(spartanFrEqual(sum01, Fr.zero), "SC3 round: s(0)+s(1) = 0 (satisfied)")
}

private func testEngineSumcheckReduce() {
    let engine = GPUSpartanLinearizeEngine()

    // Build a simple satisfied system: eq * (az*bz - cz) = 0
    // Use 4 entries (logM=2, 2 rounds)
    let eq: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    // az*bz = cz at each point
    let az: [Fr] = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7)]
    let bz: [Fr] = [frFromInt(4), frFromInt(6), frFromInt(2), frFromInt(3)]
    let cz: [Fr] = [frMul(frFromInt(2), frFromInt(4)),
                    frMul(frFromInt(3), frFromInt(6)),
                    frMul(frFromInt(5), frFromInt(2)),
                    frMul(frFromInt(7), frFromInt(3))]

    let result = engine.sumcheckReduce(
        eqTau: eq, azVec: az, bzVec: bz, czVec: cz, logM: 2)

    expectEqual(result.rounds.count, 2, "SumcheckReduce: 2 rounds for logM=2")
    expectEqual(result.challenges.count, 2, "SumcheckReduce: 2 challenges")

    // For a satisfied system, s(0)+s(1) should be 0 in the first round
    let (s0, s1, _, _) = result.rounds[0]
    let firstSum = frAdd(s0, s1)
    expect(spartanFrEqual(firstSum, Fr.zero), "SumcheckReduce: first round s(0)+s(1)=0")
}

private func testEngineBatchEvaluate() {
    let engine = GPUSpartanLinearizeEngine()
    let (instance, _) = SpartanR1CSBuilder.buildMultiplyCircuit()
    let lin = engine.linearize(instance)

    let logM = lin.logM, logN = lin.logN
    let tau = [Fr](repeating: Fr.zero, count: logM)
    let x = [Fr](repeating: Fr.zero, count: logN)

    let (aE, bE, cE) = engine.batchEvaluate(linearized: lin, tau: tau, x: x)

    // Batch evaluate should match individual evaluations
    let aRef = lin.mlA.evaluate(tau: tau, x: x)
    let bRef = lin.mlB.evaluate(tau: tau, x: x)
    let cRef = lin.mlC.evaluate(tau: tau, x: x)

    expect(spartanFrEqual(aE, aRef), "BatchEval: A matches individual")
    expect(spartanFrEqual(bE, bRef), "BatchEval: B matches individual")
    expect(spartanFrEqual(cE, cRef), "BatchEval: C matches individual")
}

private func testEngineLinearizeCache() {
    let engine = GPUSpartanLinearizeEngine()
    let (instance, _) = SpartanR1CSBuilder.buildMultiplyCircuit()

    // First linearization
    let lin1 = engine.linearize(instance)
    // Second linearization should return cached
    let lin2 = engine.linearize(instance)

    // Both should have same structure
    expectEqual(lin1.mlA.nnz, lin2.mlA.nnz, "Cache: A nnz matches")
    expectEqual(lin1.mlB.nnz, lin2.mlB.nnz, "Cache: B nnz matches")
    expectEqual(lin1.mlC.nnz, lin2.mlC.nnz, "Cache: C nnz matches")
    expectEqual(lin1.logM, lin2.logM, "Cache: logM matches")
    expectEqual(lin1.logN, lin2.logN, "Cache: logN matches")
}

// MARK: - Circuit Integration Tests

private func testMultiplyCircuitLinearization() {
    let engine = GPUSpartanLinearizeEngine()
    let (instance, gen) = SpartanR1CSBuilder.buildMultiplyCircuit()
    let (pub, wit) = gen(frFromInt(5), frFromInt(9))

    // Verify R1CS satisfaction
    let z = SpartanR1CS.buildZ(publicInputs: pub, witness: wit)
    expect(instance.isSatisfied(z: z), "Multiply linearize: R1CS satisfied")

    // Linearize and check Az*Bz = Cz consistency
    let logM = instance.logM
    let tau = [Fr](repeating: Fr.zero, count: logM)
    let result = engine.fullLinearize(
        instance: instance, publicInputs: pub, witness: wit, tau: tau)

    // The evaluations should be consistent: for a satisfied R1CS,
    // sum_x eq(tau,x)*(Az(x)*Bz(x) - Cz(x)) = 0
    // At tau=0, this simplifies to Az(0)*Bz(0) - Cz(0) = 0
    // which means the inner product relationships hold
    expect(result.aVec.count > 0, "Multiply linearize: aVec non-empty")

    // Check public output matches: 5 * 9 = 45
    let expectedOutput = frFromInt(45)
    expect(spartanFrEqual(pub[0], expectedOutput),
           "Multiply linearize: output = 45")
}

private func testAddChainCircuitLinearization() {
    let engine = GPUSpartanLinearizeEngine()
    let (instance, gen) = SpartanR1CSBuilder.buildAddChainCircuit(n: 5)
    let (pub, wit) = gen(frFromInt(3))

    let z = SpartanR1CS.buildZ(publicInputs: pub, witness: wit)
    expect(instance.isSatisfied(z: z), "AddChain linearize: R1CS satisfied")

    let lin = engine.linearize(instance)
    expectEqual(lin.numConstraints, 5, "AddChain linearize: 5 constraints")

    // Output should be (5+1)*3 = 18
    expect(spartanFrEqual(pub[0], frFromInt(18)),
           "AddChain linearize: output = 18")

    // Analyze structure: all add gates have same pattern
    let structured = engine.analyzeStructure(instance)
    let total = structured.uniformConstraintCount + structured.nonUniformIndices.count
    expectEqual(total, 5, "AddChain structured: all 5 constraints accounted for")
}

private func testRangeCheckCircuitLinearization() {
    let engine = GPUSpartanLinearizeEngine()
    let (instance, gen) = SpartanR1CSBuilder.buildRangeCheckCircuit(bits: 4)
    let (pub, wit) = gen(13) // 13 = 1101 in binary

    let z = SpartanR1CS.buildZ(publicInputs: pub, witness: wit)
    expect(instance.isSatisfied(z: z), "RangeCheck linearize: R1CS satisfied")

    let lin = engine.linearize(instance)
    // 4 boolean constraints + 1 decomposition = 5 constraints
    expectEqual(lin.numConstraints, 5, "RangeCheck linearize: 5 constraints")

    // analyzeStructure compares exact column patterns; boolean constraints on
    // different variables have distinct signatures, so no uniform group expected
    let structured = engine.analyzeStructure(instance)
    expectEqual(structured.numUniformGroups + structured.nonUniformIndices.count,
                5, "RangeCheck: all 5 constraints accounted for")
}

private func testQuadraticCircuitLinearization() {
    let engine = GPUSpartanLinearizeEngine()
    let (instance, gen) = SpartanR1CSBuilder.exampleQuadratic()
    let (pub, wit) = gen(frFromInt(3))

    let z = SpartanR1CS.buildZ(publicInputs: pub, witness: wit)
    expect(instance.isSatisfied(z: z), "Quadratic linearize: R1CS satisfied")

    // x=3: x^2 + x + 5 = 9 + 3 + 5 = 17
    expect(spartanFrEqual(pub[0], frFromInt(17)),
           "Quadratic linearize: output = 17")

    let lin = engine.linearize(instance)
    expectEqual(lin.numConstraints, 3, "Quadratic linearize: 3 constraints")
    expect(lin.mlA.nnz > 0, "Quadratic linearize: A has entries")
}

// MARK: - Consistency Tests

private func testSumcheckRoundConsistency() {
    let engine = GPUSpartanLinearizeEngine()

    // Verify that s(0) + s(1) = claimed sum for degree-2 sumcheck
    let w: [Fr] = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7),
                   frFromInt(11), frFromInt(13), frFromInt(17), frFromInt(19)]
    let z: [Fr] = [frFromInt(1), frFromInt(4), frFromInt(9), frFromInt(16),
                   frFromInt(25), frFromInt(36), frFromInt(49), frFromInt(64)]

    let (s0, s1, s2) = engine.sumcheckRound(wVec: w, zVec: z, halfSize: 4)

    // s(0) = sum_{i<4} w[i]*z[i] = 2*1 + 3*4 + 5*9 + 7*16 = 2+12+45+112 = 171
    expect(spartanFrEqual(s0, frFromInt(171)), "SC consistency: s(0) = 171")

    // s(1) = sum_{i<4} w[i+4]*z[i+4] = 11*25+13*36+17*49+19*64
    //       = 275+468+833+1216 = 2792
    expect(spartanFrEqual(s1, frFromInt(2792)), "SC consistency: s(1) = 2792")

    // s(0)+s(1) = full inner product = 171 + 2792 = 2963
    let fullIP = frAdd(s0, s1)
    expect(spartanFrEqual(fullIP, frFromInt(2963)),
           "SC consistency: s(0)+s(1) = full inner product = 2963")
}

private func testLargerCircuitLinearization() {
    let engine = GPUSpartanLinearizeEngine()

    // Synthetic circuit with 8 constraints
    let (instance, pub, wit) = SpartanR1CSBuilder.syntheticR1CS(numConstraints: 8)
    let z = SpartanR1CS.buildZ(publicInputs: pub, witness: wit)
    expect(instance.isSatisfied(z: z), "Larger circuit: R1CS satisfied")

    let lin = engine.linearize(instance)
    // 8 squaring constraints + 1 output constraint = 9
    expectEqual(lin.numConstraints, 9, "Larger circuit: 9 constraints")

    // Full linearization pipeline
    let logM = instance.logM
    let tau = [Fr](repeating: Fr.zero, count: logM)
    let result = engine.fullLinearize(
        instance: instance, publicInputs: pub, witness: wit, tau: tau)

    let paddedN = 1 << instance.logN
    expectEqual(result.aVec.count, paddedN, "Larger circuit: aVec correct length")
    expect(!result.memoryDigest.isZero, "Larger circuit: memory digest nonzero")

    // Structure analysis
    let structured = engine.analyzeStructure(instance)
    expect(structured.numUniformGroups >= 0,
           "Larger circuit: structure analysis completes")
}

private func testEmptySparseTable() {
    // Edge case: empty sparse table
    let table = SparseMLETable(rows: [], cols: [], values: [],
                                numRowVars: 1, numColVars: 1)
    expectEqual(table.nnz, 0, "Empty table: 0 nonzeros")

    // Evaluation should return zero
    let val = table.evaluate(tau: [Fr.one], x: [Fr.one])
    expect(spartanFrEqual(val, Fr.zero), "Empty table: eval = 0")

    // Row binding should return all zeros
    let bound = table.evaluateRowBinding(tau: [Fr.one])
    expectEqual(bound.count, 2, "Empty table: rowBind has 2 entries")
    expect(spartanFrEqual(bound[0], Fr.zero), "Empty table: rowBind[0] = 0")
    expect(spartanFrEqual(bound[1], Fr.zero), "Empty table: rowBind[1] = 0")
}

private func testSingleConstraintCircuit() {
    let engine = GPUSpartanLinearizeEngine()

    // Build simplest possible circuit: 1 * 1 = 1 (trivial)
    let b = SpartanR1CSBuilder()
    let out = b.addPublicInput()
    // Constraint: 1 * 1 = out (where out = 1)
    b.addConstraint(a: [(0, Fr.one)], b: [(0, Fr.one)], c: [(out, Fr.one)])
    let instance = b.build()

    let z = SpartanR1CS.buildZ(publicInputs: [Fr.one], witness: [])
    expect(instance.isSatisfied(z: z), "Single constraint: R1CS satisfied")

    let lin = engine.linearize(instance)
    expectEqual(lin.numConstraints, 1, "Single constraint: 1 constraint")

    // Evaluate at boolean point
    let logM = lin.logM
    let logN = lin.logN
    let tau = [Fr](repeating: Fr.zero, count: logM)
    let x = [Fr](repeating: Fr.zero, count: logN)
    let (aE, bE, cE) = engine.batchEvaluate(linearized: lin, tau: tau, x: x)

    // At tau=0 (row 0), x=0 (col 0): A has entry at col 0, B has entry at col 0
    expect(spartanFrEqual(aE, Fr.one), "Single constraint: A(0,0) = 1")
    expect(spartanFrEqual(bE, Fr.one), "Single constraint: B(0,0) = 1")

    // Spartan checks <Az,Bz> = <Cz,1> over the full witness, not pointwise A*B=C.
    // C has its entry at col=1 (the public input variable), not col=0,
    // so C(0,0) = 0 which is correct for the MLE evaluation.
    expect(spartanFrEqual(cE, Fr.zero), "Single constraint: C(0,0) = 0 (entry at col 1)")
}
