// GPU Nova Fold Engine Tests — folding correctness, multi-step IVC, accumulator checks
import zkMetal

// MARK: - Test Helpers

/// Build a squaring circuit: w * w = y
/// Variables: z = [1, x, y, w] where x is public input, y is public output, w = x
/// Constraint: w * w = y => A=[0,0,0,1], B=[0,0,0,1], C=[0,0,1,0]
/// Public: x, y (numPublic=2), Witness: w (1 element)
private func makeSquaringR1CS() -> NovaR1CSShape {
    let m = 1, n = 4, numPublic = 2
    var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
    aBuilder.set(row: 0, col: 3, value: Fr.one) // w

    var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
    bBuilder.set(row: 0, col: 3, value: Fr.one) // w

    var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
    cBuilder.set(row: 0, col: 2, value: Fr.one) // y

    return NovaR1CSShape(numConstraints: m, numVariables: n, numPublicInputs: numPublic,
                         A: aBuilder.build(), B: bBuilder.build(), C: cBuilder.build())
}

/// Create a valid squaring instance: x^2 = y
private func makeSquaringPair(_ val: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let x = frFromInt(val)
    let y = frMul(x, x)
    return (NovaR1CSInput(x: [x, y]), NovaR1CSWitness(W: [x]))
}

/// Build a multiply circuit: a * b = c
/// Variables: z = [1, c, a, b] (numPublic=1 for output c, witness: a, b)
private func makeMultiplyR1CS() -> NovaR1CSShape {
    let m = 1, n = 4, numPublic = 1
    var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
    aBuilder.set(row: 0, col: 2, value: Fr.one) // a

    var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
    bBuilder.set(row: 0, col: 3, value: Fr.one) // b

    var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
    cBuilder.set(row: 0, col: 1, value: Fr.one) // c

    return NovaR1CSShape(numConstraints: m, numVariables: n, numPublicInputs: numPublic,
                         A: aBuilder.build(), B: bBuilder.build(), C: cBuilder.build())
}

private func makeMultiplyPair(_ a: UInt64, _ b: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let fa = frFromInt(a)
    let fb = frFromInt(b)
    let fc = frMul(fa, fb)
    return (NovaR1CSInput(x: [fc]), NovaR1CSWitness(W: [fa, fb]))
}

// MARK: - Tests

public func runGPUNovaFoldTests() {
    suite("GPUNovaFold")

    // =========================================================================
    // Test 1: Single fold step — folded instance satisfies relaxed R1CS
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        let engine = GPUNovaFoldEngine(shape: shape)

        // Initialize with 3^2 = 9
        let (inst1, wit1) = makeSquaringPair(3)
        engine.initialize(instance: inst1, witness: wit1)

        // Fold 5^2 = 25
        let (inst2, wit2) = makeSquaringPair(5)
        let _ = engine.foldStep(newInstance: inst2, newWitness: wit2)

        let sat = engine.verifyAccumulator()
        expect(sat, "Single fold: accumulator should satisfy relaxed R1CS")
        expect(engine.stepCount == 2, "Step count should be 2 after init + fold")
    }

    // =========================================================================
    // Test 2: Multi-step fold (5 sequential folds)
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        let engine = GPUNovaFoldEngine(shape: shape)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        for i: UInt64 in 2...7 {
            steps.append(makeSquaringPair(i))
        }

        let (finalInst, finalWit) = engine.ivcChain(steps: steps)

        let sat = shape.satisfiesRelaxed(instance: finalInst, witness: finalWit)
        expect(sat, "5-step IVC chain should satisfy relaxed R1CS")
        expect(engine.stepCount == 6, "Step count should be 6 after 6 steps")

        // u should not be 1 anymore (accumulates random challenges)
        expect(!frEq(finalInst.u, Fr.one), "After folding, u != 1")
    }

    // =========================================================================
    // Test 3: Accumulator consistency — verify matches direct check
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        let engine = GPUNovaFoldEngine(shape: shape)

        let (inst1, wit1) = makeSquaringPair(7)
        engine.initialize(instance: inst1, witness: wit1)

        let (inst2, wit2) = makeSquaringPair(11)
        let _ = engine.foldStep(newInstance: inst2, newWitness: wit2)

        let (inst3, wit3) = makeSquaringPair(13)
        let _ = engine.foldStep(newInstance: inst3, newWitness: wit3)

        // Both verification methods should agree
        let accOk = engine.verifyAccumulator()
        let directOk = engine.verify(instance: engine.runningInstance!,
                                      witness: engine.runningWitness!)
        expect(accOk, "Accumulator check should pass")
        expect(directOk, "Direct verify should pass")
        expect(accOk == directOk, "Both verify methods should agree")
    }

    // =========================================================================
    // Test 4: Cross-term computation — zero for identical satisfying instances
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        let engine = GPUNovaFoldEngine(shape: shape)

        let (inst, wit) = makeSquaringPair(3)
        let (runInst, runWit) = shape.relax(instance: inst, witness: wit, pp: engine.pp)

        let T = engine.computeCrossTerm(
            runningInstance: runInst, runningWitness: runWit,
            newInstance: inst, newWitness: wit)

        // Cross-term for identical satisfying instances with u=1:
        //   T = Az1*Bz2 + Az2*Bz1 - u1*Cz2 - Cz1
        //   = 2*Az*Bz - 2*Cz = 2*(Az*Bz - Cz) = 0
        let allZero = T.allSatisfy { $0.isZero }
        expect(allZero, "Cross-term should be zero for identical satisfying instances")
    }

    // =========================================================================
    // Test 5: Identity fold — fold with zero witness (trivial circuit)
    // =========================================================================
    do {
        // Identity circuit: 1 * 1 = 1
        // z = [1, 1, w] where w = 1, constraint: A=[1,0,0] B=[1,0,0] C=[1,0,0]
        let m = 1, n = 3, numPublic = 1
        var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
        aBuilder.set(row: 0, col: 0, value: Fr.one)
        var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
        bBuilder.set(row: 0, col: 0, value: Fr.one)
        var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
        cBuilder.set(row: 0, col: 0, value: Fr.one)

        let shape = NovaR1CSShape(numConstraints: m, numVariables: n, numPublicInputs: numPublic,
                                  A: aBuilder.build(), B: bBuilder.build(), C: cBuilder.build())
        let engine = GPUNovaFoldEngine(shape: shape)

        // Instance with public = [1], witness = [0] (zero witness)
        // z = [1, 1, 0], check: A*z = 1, B*z = 1, C*z = 1, so 1*1 = 1 -- satisfied
        let inst = NovaR1CSInput(x: [Fr.one])
        let wit = NovaR1CSWitness(W: [Fr.zero])

        // Verify strict satisfaction first
        let strictOk = shape.satisfies(instance: inst, witness: wit)
        expect(strictOk, "Identity circuit should satisfy strict R1CS")

        // Initialize and fold with itself
        engine.initialize(instance: inst, witness: wit)
        let _ = engine.foldStep(newInstance: inst, newWitness: wit)

        let sat = engine.verifyAccumulator()
        expect(sat, "Identity fold with zero witness should produce valid accumulator")
    }

    // =========================================================================
    // Test 6: Non-mutating fold matches stateful fold
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        let engine = GPUNovaFoldEngine(shape: shape)

        let (inst1, wit1) = makeSquaringPair(4)
        let (runInst, runWit) = shape.relax(instance: inst1, witness: wit1, pp: engine.pp)

        let (inst2, wit2) = makeSquaringPair(6)

        // Non-mutating fold
        let (foldedInst, foldedWit, _) = engine.fold(
            runningInstance: runInst, runningWitness: runWit,
            newInstance: inst2, newWitness: wit2)

        let sat = shape.satisfiesRelaxed(instance: foldedInst, witness: foldedWit)
        expect(sat, "Non-mutating fold should produce valid relaxed instance")
    }

    // =========================================================================
    // Test 7: Multi-step with multiply circuit
    // =========================================================================
    do {
        let shape = makeMultiplyR1CS()
        let engine = GPUNovaFoldEngine(shape: shape)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        steps.append(makeMultiplyPair(3, 5))   // 3*5=15
        steps.append(makeMultiplyPair(7, 11))  // 7*11=77
        steps.append(makeMultiplyPair(2, 13))  // 2*13=26

        let (finalInst, finalWit) = engine.ivcChain(steps: steps)
        let sat = shape.satisfiesRelaxed(instance: finalInst, witness: finalWit)
        expect(sat, "Multiply IVC chain should satisfy relaxed R1CS")
    }

    // =========================================================================
    // Test 8: Reset clears accumulator
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        let engine = GPUNovaFoldEngine(shape: shape)

        let (inst, wit) = makeSquaringPair(3)
        engine.initialize(instance: inst, witness: wit)
        expect(engine.stepCount == 1, "Step count should be 1 after init")

        engine.reset()
        expect(engine.stepCount == 0, "Step count should be 0 after reset")
        expect(engine.runningInstance == nil, "Running instance should be nil after reset")
    }

    // =========================================================================
    // Test 9: Folded error vector is non-zero
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        let engine = GPUNovaFoldEngine(shape: shape)

        let (inst1, wit1) = makeSquaringPair(3)
        let (inst2, wit2) = makeSquaringPair(5)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        steps.append((inst1, wit1))
        steps.append((inst2, wit2))

        let (_, finalWit) = engine.ivcChain(steps: steps)
        let hasNonZeroError = finalWit.E.contains { !$0.isZero }
        expect(hasNonZeroError, "Folded error vector should be non-zero after fold")
    }

    // =========================================================================
    // Test 10: GPU inner product helper works
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        let engine = GPUNovaFoldEngine(shape: shape)

        let a = [frFromInt(2), frFromInt(3), frFromInt(5)]
        let b = [frFromInt(7), frFromInt(11), frFromInt(13)]
        // Expected: 2*7 + 3*11 + 5*13 = 14 + 33 + 65 = 112
        let result = engine.gpuFieldInnerProduct(a, b)
        let expected = frFromInt(112)
        expect(frEq(result, expected), "GPU inner product should compute 2*7+3*11+5*13 = 112")
    }
}
