// Nova Folding Tests — R1CS satisfaction, folding correctness, multi-step IVC
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

/// Create a valid squaring instance: x^2 = y
/// Returns (instance, witness) where x = val, y = val^2, w = val
private func makeSquaringPair(_ val: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let x = frFromInt(val)
    let y = frMul(x, x)
    return (NovaR1CSInput(x: [x, y]), NovaR1CSWitness(W: [x]))
}

/// Create a valid multiply instance: a * b = c
private func makeMultiplyPair(_ a: UInt64, _ b: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let fa = frFromInt(a)
    let fb = frFromInt(b)
    let fc = frMul(fa, fb)
    return (NovaR1CSInput(x: [fc]), NovaR1CSWitness(W: [fa, fb]))
}

// MARK: - Tests

public func runNovaFoldingTests() {
    suite("Nova Folding")

    // =========================================================================
    // Test 1: R1CS satisfaction — valid instance
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        let (inst, wit) = makeSquaringPair(3)  // 3^2 = 9
        let sat = shape.satisfies(instance: inst, witness: wit)
        expect(sat, "Squaring 3^2=9 should satisfy R1CS")
    }

    // =========================================================================
    // Test 2: R1CS satisfaction — invalid instance
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        // Wrong: claim 3^2 = 10 (should be 9)
        let x = frFromInt(3)
        let y = frFromInt(10)
        let inst = NovaR1CSInput(x: [x, y])
        let wit = NovaR1CSWitness(W: [x])
        let sat = shape.satisfies(instance: inst, witness: wit)
        expect(!sat, "3^2 != 10, should NOT satisfy")
    }

    // =========================================================================
    // Test 3: Relaxed R1CS satisfaction (E=0, u=1)
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        let (inst, wit) = makeSquaringPair(5)  // 5^2 = 25
        let prover = NovaFoldProver(shape: shape)
        let (relaxedInst, relaxedWit) = shape.relax(instance: inst, witness: wit, pp: prover.pp)
        let sat = shape.satisfiesRelaxed(instance: relaxedInst, witness: relaxedWit)
        expect(sat, "Relaxed 5^2=25 should satisfy (E=0, u=1)")
    }

    // =========================================================================
    // Test 4: Single fold — folded instance satisfies relaxed R1CS
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        let prover = NovaFoldProver(shape: shape)

        // Create running instance: 3^2 = 9
        let (inst1, wit1) = makeSquaringPair(3)
        let (runInst, runWit) = shape.relax(instance: inst1, witness: wit1, pp: prover.pp)

        // Create fresh instance: 5^2 = 25
        let (inst2, wit2) = makeSquaringPair(5)

        // Fold
        let (foldedInst, foldedWit, _) = prover.fold(
            runningInstance: runInst, runningWitness: runWit,
            newInstance: inst2, newWitness: wit2)

        // Check folded instance satisfies relaxed R1CS
        let sat = shape.satisfiesRelaxed(instance: foldedInst, witness: foldedWit)
        expect(sat, "Folded instance should satisfy relaxed R1CS")
    }

    // =========================================================================
    // Test 5: Fold verifier — checks fold correctness
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        let prover = NovaFoldProver(shape: shape)
        let verifier = NovaFoldVerifier(shape: shape)

        let (inst1, wit1) = makeSquaringPair(7)
        let (runInst, runWit) = shape.relax(instance: inst1, witness: wit1, pp: prover.pp)

        let (inst2, wit2) = makeSquaringPair(11)
        let (foldedInst, _, proof) = prover.fold(
            runningInstance: runInst, runningWitness: runWit,
            newInstance: inst2, newWitness: wit2)

        let ok = verifier.verify(running: runInst, new: inst2,
                                  proof: proof, claimed: foldedInst)
        expect(ok, "Verifier should accept valid fold")
    }

    // =========================================================================
    // Test 6: Invalid fold detection — wrong cross-term
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        let prover = NovaFoldProver(shape: shape)
        let verifier = NovaFoldVerifier(shape: shape)

        let (inst1, wit1) = makeSquaringPair(4)
        let (runInst, runWit) = shape.relax(instance: inst1, witness: wit1, pp: prover.pp)

        let (inst2, wit2) = makeSquaringPair(6)
        let (foldedInst, _, _) = prover.fold(
            runningInstance: runInst, runningWitness: runWit,
            newInstance: inst2, newWitness: wit2)

        // Create a fake proof with a wrong commitment to T
        let g = pointFromAffine(bn254G1Generator())
        let fakeCommitT = cPointScalarMul(g, frFromInt(999))
        let fakeProof = NovaFoldProof(commitT: fakeCommitT)

        // Verifier re-derives r from the fake commitT, so the expected u/x/commitE
        // will differ from the claimed folded instance (which used the real r)
        let ok = verifier.verify(running: runInst, new: inst2,
                                  proof: fakeProof, claimed: foldedInst)
        expect(!ok, "Verifier should reject fold with wrong cross-term commitment")
    }

    // =========================================================================
    // Test 7: Multi-step folding (5 sequential folds)
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        let prover = NovaFoldProver(shape: shape)

        // Create 6 valid instances: i^2 for i in 2..7
        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        for i: UInt64 in 2...7 {
            steps.append(makeSquaringPair(i))
        }

        // IVC chain: fold all 6 steps (1 base + 5 folds)
        let (finalInst, finalWit) = prover.ivcChain(steps: steps)

        // Final accumulated instance must still satisfy relaxed R1CS
        let sat = shape.satisfiesRelaxed(instance: finalInst, witness: finalWit)
        expect(sat, "5-step IVC chain should satisfy relaxed R1CS")

        // u should not be 1 anymore (it accumulates random challenges)
        expect(!frEq(finalInst.u, Fr.one), "After folding, u != 1")
    }

    // =========================================================================
    // Test 8: Multi-step folding with multiply circuit
    // =========================================================================
    do {
        let shape = makeMultiplyR1CS()
        let prover = NovaFoldProver(shape: shape)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        steps.append(makeMultiplyPair(3, 5))   // 3*5=15
        steps.append(makeMultiplyPair(7, 11))  // 7*11=77
        steps.append(makeMultiplyPair(2, 13))  // 2*13=26
        steps.append(makeMultiplyPair(4, 9))   // 4*9=36

        let (finalInst, finalWit) = prover.ivcChain(steps: steps)
        let sat = shape.satisfiesRelaxed(instance: finalInst, witness: finalWit)
        expect(sat, "Multiply IVC chain should satisfy relaxed R1CS")
    }

    // =========================================================================
    // Test 9: Relaxed R1CS — folded instance has non-zero error but satisfies
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        let prover = NovaFoldProver(shape: shape)

        let (inst1, wit1) = makeSquaringPair(3)
        let (runInst, runWit) = shape.relax(instance: inst1, witness: wit1, pp: prover.pp)

        let (inst2, wit2) = makeSquaringPair(5)
        let (foldedInst, foldedWit, _) = prover.fold(
            runningInstance: runInst, runningWitness: runWit,
            newInstance: inst2, newWitness: wit2)

        // The folded witness should have non-zero error
        let hasNonZeroError = foldedWit.E.contains { !$0.isZero }
        expect(hasNonZeroError, "Folded error vector should be non-zero")

        // But relaxed satisfaction should still hold
        let sat = shape.satisfiesRelaxed(instance: foldedInst, witness: foldedWit)
        expect(sat, "Relaxed satisfaction holds even with non-zero E")
    }

    // =========================================================================
    // Test 10: Cross-term is zero when both instances are identical
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        let prover = NovaFoldProver(shape: shape)

        let (inst, wit) = makeSquaringPair(3)
        let (runInst, runWit) = shape.relax(instance: inst, witness: wit, pp: prover.pp)

        // Compute cross-term with the same instance
        let T = prover.computeCrossTerm(
            runningInstance: runInst, runningWitness: runWit,
            newInstance: inst, newWitness: wit)

        // Cross-term for identical satisfying instances with u=1:
        //   T = Az1*Bz2 + Az2*Bz1 - u1*Cz2 - Cz1
        //   = 2*Az*Bz - 2*Cz = 2*(Az*Bz - Cz) = 0
        let allZero = T.allSatisfy { $0.isZero }
        expect(allZero, "Cross-term should be zero for identical satisfying instances")
    }

    // =========================================================================
    // Test 11: Verifier rejects tampered u
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        let prover = NovaFoldProver(shape: shape)
        let verifier = NovaFoldVerifier(shape: shape)

        let (inst1, wit1) = makeSquaringPair(8)
        let (runInst, runWit) = shape.relax(instance: inst1, witness: wit1, pp: prover.pp)

        let (inst2, wit2) = makeSquaringPair(12)
        let (foldedInst, _, proof) = prover.fold(
            runningInstance: runInst, runningWitness: runWit,
            newInstance: inst2, newWitness: wit2)

        // Tamper with u
        let tamperedInst = NovaRelaxedInstance(
            commitW: foldedInst.commitW,
            commitE: foldedInst.commitE,
            u: frAdd(foldedInst.u, Fr.one),  // wrong u
            x: foldedInst.x)

        let ok = verifier.verify(running: runInst, new: inst2,
                                  proof: proof, claimed: tamperedInst)
        expect(!ok, "Verifier should reject tampered u")
    }

    // =========================================================================
    // Test 12: Verifier rejects tampered public input
    // =========================================================================
    do {
        let shape = makeSquaringR1CS()
        let prover = NovaFoldProver(shape: shape)
        let verifier = NovaFoldVerifier(shape: shape)

        let (inst1, wit1) = makeSquaringPair(2)
        let (runInst, runWit) = shape.relax(instance: inst1, witness: wit1, pp: prover.pp)

        let (inst2, wit2) = makeSquaringPair(10)
        let (foldedInst, _, proof) = prover.fold(
            runningInstance: runInst, runningWitness: runWit,
            newInstance: inst2, newWitness: wit2)

        // Tamper with public input
        var badX = foldedInst.x
        badX[0] = frAdd(badX[0], Fr.one)
        let tamperedInst = NovaRelaxedInstance(
            commitW: foldedInst.commitW,
            commitE: foldedInst.commitE,
            u: foldedInst.u,
            x: badX)

        let ok = verifier.verify(running: runInst, new: inst2,
                                  proof: proof, claimed: tamperedInst)
        expect(!ok, "Verifier should reject tampered public input")
    }
}
