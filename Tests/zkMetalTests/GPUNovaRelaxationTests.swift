// GPU Nova Relaxation Engine Tests — relaxation lifecycle, error accumulation,
// scalar tracking, folding, diagnostics, batch operations, snapshot chains
import zkMetal
import Foundation

// MARK: - Test Helpers

/// Squaring circuit: w*w=y. z=[1,x,y,w], numPublic=2, witness=w
private func makeRelaxSquaringR1CS() -> NovaR1CSShape {
    let m = 1, n = 4, numPublic = 2
    var aB = SparseMatrixBuilder(rows: m, cols: n); aB.set(row: 0, col: 3, value: Fr.one)
    var bB = SparseMatrixBuilder(rows: m, cols: n); bB.set(row: 0, col: 3, value: Fr.one)
    var cB = SparseMatrixBuilder(rows: m, cols: n); cB.set(row: 0, col: 2, value: Fr.one)
    return NovaR1CSShape(numConstraints: m, numVariables: n, numPublicInputs: numPublic,
                         A: aB.build(), B: bB.build(), C: cB.build())
}

/// Create a valid squaring instance: x^2 = y
private func makeRelaxSquaringPair(_ val: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let x = frFromInt(val)
    let y = frMul(x, x)
    return (NovaR1CSInput(x: [x, y]), NovaR1CSWitness(W: [x]))
}

/// Multiply circuit: a*b=c. z=[1,c,a,b], numPublic=1, witness=a,b
private func makeRelaxMultiplyR1CS() -> NovaR1CSShape {
    let m = 1, n = 4, numPublic = 1
    var aB = SparseMatrixBuilder(rows: m, cols: n); aB.set(row: 0, col: 2, value: Fr.one)
    var bB = SparseMatrixBuilder(rows: m, cols: n); bB.set(row: 0, col: 3, value: Fr.one)
    var cB = SparseMatrixBuilder(rows: m, cols: n); cB.set(row: 0, col: 1, value: Fr.one)
    return NovaR1CSShape(numConstraints: m, numVariables: n, numPublicInputs: numPublic,
                         A: aB.build(), B: bB.build(), C: cB.build())
}

private func makeRelaxMultiplyPair(_ a: UInt64, _ b: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let fa = frFromInt(a)
    let fb = frFromInt(b)
    let fc = frMul(fa, fb)
    return (NovaR1CSInput(x: [fc]), NovaR1CSWitness(W: [fa, fb]))
}

/// Two-constraint circuit: w0*w0=y, (w0+w1)*1=x. z=[1,x,y,w0,w1]
private func makeTwoConstraintR1CS() -> NovaR1CSShape {
    let m = 2, n = 5, numPublic = 2
    var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
    aBuilder.set(row: 0, col: 3, value: Fr.one) // w0
    aBuilder.set(row: 1, col: 3, value: Fr.one) // w0
    aBuilder.set(row: 1, col: 4, value: Fr.one) // w1

    var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
    bBuilder.set(row: 0, col: 3, value: Fr.one) // w0
    bBuilder.set(row: 1, col: 0, value: Fr.one) // 1

    var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
    cBuilder.set(row: 0, col: 2, value: Fr.one) // y
    cBuilder.set(row: 1, col: 1, value: Fr.one) // x

    return NovaR1CSShape(numConstraints: m, numVariables: n, numPublicInputs: numPublic,
                         A: aBuilder.build(), B: bBuilder.build(), C: cBuilder.build())
}

/// Pair for two-constraint circuit: x=w0+w1, y=w0^2
private func makeTwoConstraintPair(_ v0: UInt64, _ v1: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let w0 = frFromInt(v0)
    let w1 = frFromInt(v1)
    let x = frAdd(w0, w1)
    let y = frMul(w0, w0)
    return (NovaR1CSInput(x: [x, y]), NovaR1CSWitness(W: [w0, w1]))
}

/// Build identity circuit: 1*1=1 (always satisfied), z = [1, p, w]
private func makeIdentityR1CS() -> NovaR1CSShape {
    let m = 1, n = 3
    var ab = SparseMatrixBuilder(rows: m, cols: n)
    ab.set(row: 0, col: 0, value: Fr.one)
    var cb = SparseMatrixBuilder(rows: m, cols: n)
    cb.set(row: 0, col: 0, value: Fr.one)
    var db = SparseMatrixBuilder(rows: m, cols: n)
    db.set(row: 0, col: 0, value: Fr.one)
    return NovaR1CSShape(numConstraints: m, numVariables: n, numPublicInputs: 1,
                         A: ab.build(), B: cb.build(), C: db.build())
}

// MARK: - Tests

public func runGPUNovaRelaxationTests() {
    suite("GPUNovaRelaxation")

    // =========================================================================
    // Test 1: Strict-to-relaxed conversion — basic properties
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst, wit) = makeRelaxSquaringPair(3)
        let (relaxedInst, relaxedWit) = engine.relaxStrict(instance: inst, witness: wit)

        // u should be 1
        expect(frEq(relaxedInst.u, Fr.one), "Relaxed instance should have u=1")

        // Error vector should be all zeros
        let allZero = relaxedWit.E.allSatisfy { $0.isZero }
        expect(allZero, "Relaxed witness should have zero error vector")

        // Error vector length should match numConstraints
        expectEqual(relaxedWit.E.count, shape.numConstraints,
                    "Error vector length should equal numConstraints")

        // Witness should be preserved
        expectEqual(relaxedWit.W.count, wit.W.count,
                    "Witness length should be preserved")
        expect(frEq(relaxedWit.W[0], wit.W[0]), "Witness values should be preserved")

        // Public input should be preserved
        expectEqual(relaxedInst.x.count, inst.x.count,
                    "Public input length should be preserved")
        for i in 0..<inst.x.count {
            expect(frEq(relaxedInst.x[i], inst.x[i]),
                   "Public input[\(i)] should be preserved")
        }
    }

    // =========================================================================
    // Test 2: Relaxed instance satisfies relaxed R1CS
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst, wit) = makeRelaxSquaringPair(7)
        let (relaxedInst, relaxedWit) = engine.relaxStrict(instance: inst, witness: wit)

        let sat = shape.satisfiesRelaxed(instance: relaxedInst, witness: relaxedWit)
        expect(sat, "Freshly relaxed instance should satisfy relaxed R1CS")
    }

    // =========================================================================
    // Test 3: initializeFromStrict sets engine state
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst, wit) = makeRelaxSquaringPair(5)
        let relaxedInst = engine.initializeFromStrict(instance: inst, witness: wit)

        expect(engine.currentInstance != nil, "Current instance should be set")
        expect(engine.currentWitness != nil, "Current witness should be set")
        expectEqual(engine.foldCount, 0, "Fold count should be 0 after init")
        expect(frEq(relaxedInst.u, Fr.one), "Initial u should be 1")
    }

    // =========================================================================
    // Test 4: Error accumulation — E' = E + r * T
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let E = [frFromInt(10)]
        let T = [frFromInt(3)]
        let r = frFromInt(7)

        let result = engine.accumulateError(E, crossTerm: T, r: r)
        // E' = 10 + 7 * 3 = 10 + 21 = 31
        let expected = frFromInt(31)
        expect(frEq(result[0], expected), "accumulateError: 10 + 7*3 = 31")
    }

    // =========================================================================
    // Test 5: Error accumulation from zero — E' = 0 + r * T = r * T
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let E = [Fr.zero]
        let T = [frFromInt(13)]
        let r = frFromInt(5)

        let result = engine.accumulateError(E, crossTerm: T, r: r)
        let expected = frFromInt(65) // 0 + 5*13 = 65
        expect(frEq(result[0], expected), "accumulateError from zero: 0 + 5*13 = 65")
    }

    // =========================================================================
    // Test 6: Scalar update — u' = u + r
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let u = Fr.one
        let r = frFromInt(42)
        let result = engine.updateScalar(u, r: r)
        let expected = frFromInt(43) // 1 + 42 = 43
        expect(frEq(result, expected), "updateScalar: 1 + 42 = 43")
    }

    // =========================================================================
    // Test 7: Public input fold — x' = x1 + r * x2
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let x1 = [frFromInt(10), frFromInt(20)]
        let x2 = [frFromInt(3), frFromInt(7)]
        let r = frFromInt(5)

        let result = engine.foldPublicInput(x1, x2, r: r)
        // x'[0] = 10 + 5*3 = 25, x'[1] = 20 + 5*7 = 55
        expect(frEq(result[0], frFromInt(25)), "foldPublicInput[0]: 10 + 5*3 = 25")
        expect(frEq(result[1], frFromInt(55)), "foldPublicInput[1]: 20 + 5*7 = 55")
    }

    // =========================================================================
    // Test 8: Witness fold — W' = W1 + r * W2
    // =========================================================================
    do {
        let shape = makeRelaxMultiplyR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let w1 = [frFromInt(4), frFromInt(6)]
        let w2 = [frFromInt(2), frFromInt(3)]
        let r = frFromInt(10)

        let result = engine.foldWitness(w1, w2, r: r)
        // W'[0] = 4 + 10*2 = 24, W'[1] = 6 + 10*3 = 36
        expect(frEq(result[0], frFromInt(24)), "foldWitness[0]: 4 + 10*2 = 24")
        expect(frEq(result[1], frFromInt(36)), "foldWitness[1]: 6 + 10*3 = 36")
    }

    // =========================================================================
    // Test 9: Cross-term for identical satisfying instances is zero
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst, wit) = makeRelaxSquaringPair(3)
        let (ri, rw) = engine.relaxStrict(instance: inst, witness: wit)

        let T = engine.computeCrossTerm(inst1: ri, wit1: rw, inst2: ri, wit2: rw)
        let allZero = T.allSatisfy { $0.isZero }
        expect(allZero, "Cross-term for identical relaxed instances should be zero")
    }

    // =========================================================================
    // Test 10: Cross-term strict — identical instances produce zero T
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst, wit) = makeRelaxSquaringPair(3)
        let (ri, rw) = engine.relaxStrict(instance: inst, witness: wit)

        let T = engine.computeCrossTermStrict(
            relaxedInst: ri, relaxedWit: rw,
            strictInst: inst, strictWit: wit)
        let allZero = T.allSatisfy { $0.isZero }
        expect(allZero, "Cross-term strict for identical instances should be zero")
    }

    // =========================================================================
    // Test 11: Cross-term for different instances is non-zero
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst1, wit1) = makeRelaxSquaringPair(3)
        let (inst2, wit2) = makeRelaxSquaringPair(7)
        let (ri, rw) = engine.relaxStrict(instance: inst1, witness: wit1)

        let T = engine.computeCrossTermStrict(
            relaxedInst: ri, relaxedWit: rw,
            strictInst: inst2, strictWit: wit2)
        let hasNonZero = T.contains { !$0.isZero }
        expect(hasNonZero, "Cross-term for different instances should be non-zero")
    }

    // =========================================================================
    // Test 12: foldStrictIntoRunning — single fold produces valid accumulator
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst1, wit1) = makeRelaxSquaringPair(3)
        engine.initializeFromStrict(instance: inst1, witness: wit1)

        let (inst2, wit2) = makeRelaxSquaringPair(5)
        let _ = engine.foldStrictIntoRunning(instance: inst2, witness: wit2)

        let sat = engine.verifyCurrentInstance()
        expect(sat, "Single fold into running should satisfy relaxed R1CS")
        expectEqual(engine.foldCount, 1, "Fold count should be 1 after one fold")
    }

    // =========================================================================
    // Test 13: Multi-step foldStrictIntoRunning — 5 folds
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst0, wit0) = makeRelaxSquaringPair(2)
        engine.initializeFromStrict(instance: inst0, witness: wit0)

        for i: UInt64 in 3...7 {
            let (inst, wit) = makeRelaxSquaringPair(i)
            engine.foldStrictIntoRunning(instance: inst, witness: wit)
        }

        let sat = engine.verifyCurrentInstance()
        expect(sat, "5-fold chain should satisfy relaxed R1CS")
        expectEqual(engine.foldCount, 5, "Fold count should be 5")

        // u should not be 1 after folding
        expect(!frEq(engine.currentInstance!.u, Fr.one),
               "After 5 folds, u should not be 1")
    }

    // =========================================================================
    // Test 14: Error vector is non-zero after folding different instances
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst1, wit1) = makeRelaxSquaringPair(3)
        engine.initializeFromStrict(instance: inst1, witness: wit1)

        let (inst2, wit2) = makeRelaxSquaringPair(11)
        engine.foldStrictIntoRunning(instance: inst2, witness: wit2)

        let hasNonZeroE = engine.currentWitness!.E.contains { !$0.isZero }
        expect(hasNonZeroE, "Error vector should be non-zero after fold")
    }

    // =========================================================================
    // Test 15: Diagnostic on freshly relaxed instance
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst, wit) = makeRelaxSquaringPair(4)
        let (ri, rw) = engine.relaxStrict(instance: inst, witness: wit)

        let diag = engine.diagnose(instance: ri, witness: rw)
        expect(diag.satisfied, "Freshly relaxed instance should be diagnosed as satisfied")
        expect(diag.failingConstraints.isEmpty, "No failing constraints expected")
        expect(frEq(diag.u, Fr.one), "Diagnostic u should be 1")
        expect(diag.errorIsZero, "Error should be zero for freshly relaxed instance")
        expectEqual(diag.numConstraints, shape.numConstraints, "numConstraints should match shape")
    }

    // =========================================================================
    // Test 16: Diagnostic on folded instance
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst1, wit1) = makeRelaxSquaringPair(3)
        engine.initializeFromStrict(instance: inst1, witness: wit1)

        let (inst2, wit2) = makeRelaxSquaringPair(7)
        engine.foldStrictIntoRunning(instance: inst2, witness: wit2)

        let diag = engine.diagnoseCurrent()!
        expect(diag.satisfied, "Folded instance should be diagnosed as satisfied")
        expect(!diag.errorIsZero, "Error should not be zero after fold")
        expect(!frEq(diag.u, Fr.one), "u should not be 1 after fold")
    }

    // =========================================================================
    // Test 17: Batch relaxation
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        var pairs = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        for i: UInt64 in 2...6 {
            pairs.append(makeRelaxSquaringPair(i))
        }

        let relaxed = engine.batchRelax(pairs: pairs)
        expectEqual(relaxed.count, 5, "Batch relax should produce 5 pairs")

        for (idx, (ri, rw)) in relaxed.enumerated() {
            let sat = shape.satisfiesRelaxed(instance: ri, witness: rw)
            expect(sat, "Batch relaxed[\(idx)] should satisfy relaxed R1CS")
            expect(frEq(ri.u, Fr.one), "Batch relaxed[\(idx)] should have u=1")
        }
    }

    // =========================================================================
    // Test 18: Error vector analysis — non-zero count and density
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        // Zero error vector
        let zeroE = [Fr.zero, Fr.zero, Fr.zero]
        expectEqual(engine.errorNonZeroCount(zeroE), 0, "Zero vector has 0 non-zero entries")
        expect(engine.errorIsZero(zeroE), "Zero vector should be identified as zero")
        expect(engine.errorDensity(zeroE) == 0.0, "Zero vector density should be 0.0")

        // Non-zero error vector
        let nonZeroE = [frFromInt(1), Fr.zero, frFromInt(5)]
        expectEqual(engine.errorNonZeroCount(nonZeroE), 2, "Should have 2 non-zero entries")
        expect(!engine.errorIsZero(nonZeroE), "Non-zero vector should not be zero")

        // Empty vector
        let emptyE = [Fr]()
        expect(engine.errorIsZero(emptyE), "Empty vector should be zero")
        expect(engine.errorDensity(emptyE) == 0.0, "Empty vector density should be 0.0")
    }

    // =========================================================================
    // Test 19: Scalar chain analysis — expected scalar from challenges
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        // No challenges -> u = 1
        let u0 = engine.expectedScalar(challenges: [])
        expect(frEq(u0, Fr.one), "No challenges should give u=1")

        // Single challenge r=5 -> u = 1 + 5 = 6
        let u1 = engine.expectedScalar(challenges: [frFromInt(5)])
        expect(frEq(u1, frFromInt(6)), "One challenge r=5 should give u=6")

        // Two challenges r=3, r=7 -> u = 1 + 3 + 7 = 11
        let u2 = engine.expectedScalar(challenges: [frFromInt(3), frFromInt(7)])
        expect(frEq(u2, frFromInt(11)), "Two challenges should give u=11")
    }

    // =========================================================================
    // Test 20: Scalar consistency check after multi-step fold
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)
        engine.recordErrorHistory = true

        let (inst0, wit0) = makeRelaxSquaringPair(2)
        engine.initializeFromStrict(instance: inst0, witness: wit0)

        for i: UInt64 in 3...5 {
            let (inst, wit) = makeRelaxSquaringPair(i)
            engine.foldStrictIntoRunning(instance: inst, witness: wit)
        }

        let consistent = engine.verifyScalarConsistency()
        expect(consistent, "Scalar u should be consistent with recorded challenges")
    }

    // =========================================================================
    // Test 21: Snapshot recording
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)
        engine.recordSnapshots = true

        let (inst0, wit0) = makeRelaxSquaringPair(3)
        engine.initializeFromStrict(instance: inst0, witness: wit0)

        for i: UInt64 in 4...6 {
            let (inst, wit) = makeRelaxSquaringPair(i)
            engine.foldStrictIntoRunning(instance: inst, witness: wit)
        }

        // Should have 4 snapshots: init + 3 folds
        expectEqual(engine.snapshots.count, 4, "Should have 4 snapshots")
        expectEqual(engine.snapshots[0].stepIndex, 0, "First snapshot step should be 0")
        expectEqual(engine.snapshots[1].stepIndex, 1, "Second snapshot step should be 1")
        expectEqual(engine.snapshots[2].stepIndex, 2, "Third snapshot step should be 2")
        expectEqual(engine.snapshots[3].stepIndex, 3, "Fourth snapshot step should be 3")

        // First snapshot u should be 1
        expect(frEq(engine.snapshots[0].u, Fr.one), "Initial snapshot u should be 1")

        // Last snapshot u should match current instance
        expect(frEq(engine.snapshots.last!.u, engine.currentInstance!.u),
               "Last snapshot u should match current instance u")
    }

    // =========================================================================
    // Test 22: Snapshot chain verification
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)
        engine.recordSnapshots = true

        let (inst0, wit0) = makeRelaxSquaringPair(3)
        engine.initializeFromStrict(instance: inst0, witness: wit0)

        let (inst1, wit1) = makeRelaxSquaringPair(5)
        engine.foldStrictIntoRunning(instance: inst1, witness: wit1)

        let chainOk = engine.verifySnapshotChain()
        expect(chainOk, "Snapshot chain should be valid")
    }

    // =========================================================================
    // Test 23: Error accumulation records
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)
        engine.recordErrorHistory = true

        let (inst0, wit0) = makeRelaxSquaringPair(2)
        engine.initializeFromStrict(instance: inst0, witness: wit0)

        let (inst1, wit1) = makeRelaxSquaringPair(5)
        engine.foldStrictIntoRunning(instance: inst1, witness: wit1)

        let (inst2, wit2) = makeRelaxSquaringPair(7)
        engine.foldStrictIntoRunning(instance: inst2, witness: wit2)

        expectEqual(engine.errorRecords.count, 2, "Should have 2 error records")
        expectEqual(engine.errorRecords[0].stepIndex, 1, "First record step should be 1")
        expectEqual(engine.errorRecords[1].stepIndex, 2, "Second record step should be 2")

        // Challenge scalars should be non-zero field elements
        expect(!engine.errorRecords[0].challenge.isZero,
               "First challenge should be non-zero")
        expect(!engine.errorRecords[1].challenge.isZero,
               "Second challenge should be non-zero")
    }

    // =========================================================================
    // Test 24: Reset clears all state
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst, wit) = makeRelaxSquaringPair(3)
        engine.initializeFromStrict(instance: inst, witness: wit)
        engine.foldStrictIntoRunning(
            instance: makeRelaxSquaringPair(5).0,
            witness: makeRelaxSquaringPair(5).1)

        engine.reset()

        expect(engine.currentInstance == nil, "Instance should be nil after reset")
        expect(engine.currentWitness == nil, "Witness should be nil after reset")
        expectEqual(engine.foldCount, 0, "Fold count should be 0 after reset")
        expectEqual(engine.snapshots.count, 0, "Snapshots should be empty after reset")
        expectEqual(engine.errorRecords.count, 0, "Error records should be empty after reset")
    }

    // =========================================================================
    // Test 25: Fold relaxed pair — both relaxed instances fold correctly
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst1, wit1) = makeRelaxSquaringPair(3)
        let (inst2, wit2) = makeRelaxSquaringPair(5)
        let (ri1, rw1) = engine.relaxStrict(instance: inst1, witness: wit1)
        let (ri2, rw2) = engine.relaxStrict(instance: inst2, witness: wit2)

        let result = engine.foldRelaxedPair(inst1: ri1, wit1: rw1, inst2: ri2, wit2: rw2)

        let sat = shape.satisfiesRelaxed(instance: result.instance, witness: result.witness)
        expect(sat, "Folded relaxed pair should satisfy relaxed R1CS")
        expect(!result.challenge.isZero, "Fold challenge should be non-zero")
    }

    // =========================================================================
    // Test 26: Fold relaxed pair — cross-term and commitment are produced
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst1, wit1) = makeRelaxSquaringPair(3)
        let (inst2, wit2) = makeRelaxSquaringPair(7)
        let (ri1, rw1) = engine.relaxStrict(instance: inst1, witness: wit1)
        let (ri2, rw2) = engine.relaxStrict(instance: inst2, witness: wit2)

        let result = engine.foldRelaxedPair(inst1: ri1, wit1: rw1, inst2: ri2, wit2: rw2)

        expectEqual(result.crossTerm.count, shape.numConstraints,
                    "Cross-term length should match numConstraints")
        // commitT should not be identity if T is non-zero for different instances
        // (can be zero for identical instances, but 3 != 7 so it should be non-zero)
    }

    // =========================================================================
    // Test 27: Multiply circuit — relaxation and folding
    // =========================================================================
    do {
        let shape = makeRelaxMultiplyR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst0, wit0) = makeRelaxMultiplyPair(3, 5)
        engine.initializeFromStrict(instance: inst0, witness: wit0)

        let (inst1, wit1) = makeRelaxMultiplyPair(7, 11)
        engine.foldStrictIntoRunning(instance: inst1, witness: wit1)

        let (inst2, wit2) = makeRelaxMultiplyPair(2, 13)
        engine.foldStrictIntoRunning(instance: inst2, witness: wit2)

        let sat = engine.verifyCurrentInstance()
        expect(sat, "Multiply circuit fold chain should satisfy relaxed R1CS")
        expectEqual(engine.foldCount, 2, "Should have 2 fold steps")
    }

    // =========================================================================
    // Test 28: Two-constraint circuit — relaxation
    // =========================================================================
    do {
        let shape = makeTwoConstraintR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst, wit) = makeTwoConstraintPair(3, 4) // w0=3, w1=4, x=7, y=9
        let (ri, rw) = engine.relaxStrict(instance: inst, witness: wit)

        // Verify strict satisfaction first
        let strictSat = shape.satisfies(instance: inst, witness: wit)
        expect(strictSat, "Two-constraint circuit should satisfy strict R1CS")

        // Verify relaxed satisfaction
        let relaxedSat = shape.satisfiesRelaxed(instance: ri, witness: rw)
        expect(relaxedSat, "Two-constraint relaxed instance should satisfy relaxed R1CS")

        expectEqual(rw.E.count, 2, "Error vector should have 2 entries (numConstraints=2)")
    }

    // =========================================================================
    // Test 29: Two-constraint circuit — multi-step fold
    // =========================================================================
    do {
        let shape = makeTwoConstraintR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst0, wit0) = makeTwoConstraintPair(2, 3)
        engine.initializeFromStrict(instance: inst0, witness: wit0)

        let (inst1, wit1) = makeTwoConstraintPair(5, 7)
        engine.foldStrictIntoRunning(instance: inst1, witness: wit1)

        let (inst2, wit2) = makeTwoConstraintPair(11, 13)
        engine.foldStrictIntoRunning(instance: inst2, witness: wit2)

        let sat = engine.verifyCurrentInstance()
        expect(sat, "Two-constraint 3-step fold should satisfy relaxed R1CS")

        let diag = engine.diagnoseCurrent()!
        expect(diag.satisfied, "Diagnostic should report satisfied")
        expectEqual(diag.numConstraints, 2, "Should have 2 constraints")
    }

    // =========================================================================
    // Test 30: Identity circuit — relaxation and fold
    // =========================================================================
    do {
        let shape = makeIdentityR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let inst = NovaR1CSInput(x: [Fr.one])
        let wit = NovaR1CSWitness(W: [Fr.zero])

        let strictSat = shape.satisfies(instance: inst, witness: wit)
        expect(strictSat, "Identity circuit should satisfy strict R1CS")

        let (ri, rw) = engine.relaxStrict(instance: inst, witness: wit)
        let relaxedSat = shape.satisfiesRelaxed(instance: ri, witness: rw)
        expect(relaxedSat, "Identity relaxed instance should satisfy relaxed R1CS")

        // Fold with itself
        engine.initializeFromStrict(instance: inst, witness: wit)
        engine.foldStrictIntoRunning(instance: inst, witness: wit)
        let foldSat = engine.verifyCurrentInstance()
        expect(foldSat, "Identity circuit fold should satisfy relaxed R1CS")
    }

    // =========================================================================
    // Test 32: GPU matrix-vector and inner product
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        // MatVec: check GPU matches CPU
        let z = [Fr.one, frFromInt(3), frFromInt(9), frFromInt(3)]
        let gpuAz = engine.gpuMatVec(shape.A, z)
        let cpuAz = shape.A.mulVec(z)
        for i in 0..<gpuAz.count {
            expect(frEq(gpuAz[i], cpuAz[i]), "GPU/CPU A*z[\(i)] should match")
        }

        // Inner product: 2*7 + 3*11 + 5*13 = 112
        let a = [frFromInt(2), frFromInt(3), frFromInt(5)]
        let b = [frFromInt(7), frFromInt(11), frFromInt(13)]
        let ip = engine.gpuFieldInnerProduct(a, b)
        expect(frEq(ip, frFromInt(112)), "Inner product should be 112")
    }

    // =========================================================================
    // Test 33: Disable snapshot and error history recording
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let a = [frFromInt(2), frFromInt(3), frFromInt(5)]
        let b = [frFromInt(7), frFromInt(11), frFromInt(13)]
        // Expected: 2*7 + 3*11 + 5*13 = 14 + 33 + 65 = 112
        let result = engine.gpuFieldInnerProduct(a, b)
        let expected = frFromInt(112)
        expect(frEq(result, expected), "Inner product should be 112")
    }

    // =========================================================================
    // Test 34: Disable recording, uninitialized state
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        // Uninitialized checks
        expect(!engine.verifyCurrentInstance(), "verify should return false before init")
        expect(engine.diagnoseCurrent() == nil, "diagnose should return nil before init")

        // Disable recording, then fold
        engine.recordSnapshots = false
        engine.recordErrorHistory = false

        let (inst0, wit0) = makeRelaxSquaringPair(3)
        engine.initializeFromStrict(instance: inst0, witness: wit0)
        let (inst1, wit1) = makeRelaxSquaringPair(5)
        engine.foldStrictIntoRunning(instance: inst1, witness: wit1)

        expectEqual(engine.snapshots.count, 0, "Snapshots should be empty when disabled")
        expectEqual(engine.errorRecords.count, 0, "Error records should be empty when disabled")
    }

    // =========================================================================
    // Test 39: Re-initialization resets state completely
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst0, wit0) = makeRelaxSquaringPair(3)
        engine.initializeFromStrict(instance: inst0, witness: wit0)
        let (inst1, wit1) = makeRelaxSquaringPair(5)
        engine.foldStrictIntoRunning(instance: inst1, witness: wit1)

        let u1 = engine.currentInstance!.u

        // Re-initialize
        let (inst2, wit2) = makeRelaxSquaringPair(7)
        engine.initializeFromStrict(instance: inst2, witness: wit2)

        expectEqual(engine.foldCount, 0, "Re-init should reset fold count")
        expect(frEq(engine.currentInstance!.u, Fr.one), "Re-init should reset u to 1")
        expect(!frEq(engine.currentInstance!.u, u1), "Re-init u should differ from first chain u")
    }

    // =========================================================================
    // Test 40: Long fold chain (10 steps) — stress test
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)
        engine.recordSnapshots = true
        engine.recordErrorHistory = true

        let (inst0, wit0) = makeRelaxSquaringPair(2)
        engine.initializeFromStrict(instance: inst0, witness: wit0)

        for i: UInt64 in 3...12 {
            let (inst, wit) = makeRelaxSquaringPair(i)
            engine.foldStrictIntoRunning(instance: inst, witness: wit)
        }

        let sat = engine.verifyCurrentInstance()
        expect(sat, "10-step fold chain should satisfy relaxed R1CS")
        expectEqual(engine.foldCount, 10, "Should have 10 folds")
        expectEqual(engine.snapshots.count, 11, "Should have 11 snapshots (init + 10)")
        expectEqual(engine.errorRecords.count, 10, "Should have 10 error records")

        let chainOk = engine.verifySnapshotChain()
        expect(chainOk, "10-step snapshot chain should be valid")

        let scalarOk = engine.verifyScalarConsistency()
        expect(scalarOk, "10-step scalar consistency should hold")
    }

    // =========================================================================
    // Test 42: Version, GPU flag, cpuThreshold
    // =========================================================================
    do {
        let ver = GPUNovaRelaxationEngine.version
        expect(!ver.version.isEmpty, "Version string should not be empty")

        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)
        let _ = engine.gpuAvailable
        expectEqual(engine.cpuThreshold, 512, "Default cpuThreshold should be 512")
        engine.cpuThreshold = 1024
        expectEqual(engine.cpuThreshold, 1024, "cpuThreshold should be settable to 1024")
    }

    // =========================================================================
    // Test 43: Fold relaxed pair with pre-folded instances
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        // Create two separate fold chains, then fold their results together
        let (inst1, wit1) = makeRelaxSquaringPair(3)
        let (inst2, wit2) = makeRelaxSquaringPair(5)
        let (ri1, rw1) = engine.relaxStrict(instance: inst1, witness: wit1)
        let (ri2, rw2) = engine.relaxStrict(instance: inst2, witness: wit2)

        // First fold: relax(3) + relax(5)
        let result1 = engine.foldRelaxedPair(inst1: ri1, wit1: rw1, inst2: ri2, wit2: rw2)
        let sat1 = shape.satisfiesRelaxed(instance: result1.instance, witness: result1.witness)
        expect(sat1, "First relaxed pair fold should satisfy relaxed R1CS")

        // Create another relaxed instance
        let (inst3, wit3) = makeRelaxSquaringPair(7)
        let (ri3, rw3) = engine.relaxStrict(instance: inst3, witness: wit3)

        // Second fold: result1 + relax(7)
        let result2 = engine.foldRelaxedPair(
            inst1: result1.instance, wit1: result1.witness,
            inst2: ri3, wit2: rw3)
        let sat2 = shape.satisfiesRelaxed(instance: result2.instance, witness: result2.witness)
        expect(sat2, "Second relaxed pair fold should satisfy relaxed R1CS")
    }

    // =========================================================================
    // Test 45: Fold preserves vector lengths
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let engine = GPUNovaRelaxationEngine(shape: shape)

        let (inst0, wit0) = makeRelaxSquaringPair(3)
        engine.initializeFromStrict(instance: inst0, witness: wit0)

        let initialXLen = engine.currentInstance!.x.count
        let initialWLen = engine.currentWitness!.W.count

        for i: UInt64 in 4...6 {
            let (inst, wit) = makeRelaxSquaringPair(i)
            engine.foldStrictIntoRunning(instance: inst, witness: wit)
            expectEqual(engine.currentInstance!.x.count, initialXLen,
                        "Public input length should be preserved")
            expectEqual(engine.currentWitness!.W.count, initialWLen,
                        "Witness length should be preserved")
            expectEqual(engine.currentWitness!.E.count, shape.numConstraints,
                        "Error vector length should match numConstraints")
        }
    }

    // =========================================================================
    // Test 46: Custom Pedersen params constructor
    // =========================================================================
    do {
        let shape = makeRelaxSquaringR1CS()
        let ppW = PedersenParams.generate(size: max(shape.numWitness, 1))
        let ppE = PedersenParams.generate(size: max(shape.numConstraints, 1))
        let engine = GPUNovaRelaxationEngine(shape: shape, ppW: ppW, ppE: ppE)

        let (inst, wit) = makeRelaxSquaringPair(3)
        engine.initializeFromStrict(instance: inst, witness: wit)

        let sat = engine.verifyCurrentInstance()
        expect(sat, "Engine with custom PP should produce valid relaxed instance")
    }
}
