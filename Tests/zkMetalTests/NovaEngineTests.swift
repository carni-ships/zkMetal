// Nova Engine Tests — IVC prover/verifier correctness, multi-step chains, rejection
import zkMetal

// MARK: - Test Helpers

/// Build a squaring circuit: w * w = y
/// Variables: z = [1, x, y, w] where x is public input, y is public output, w = x
/// Constraint: w * w = y => A=[0,0,0,1], B=[0,0,0,1], C=[0,0,1,0]
/// Public: x, y (numPublic=2), Witness: w (1 element)
private func engineSquaringShape() -> NovaR1CSShape {
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
private func engineMultiplyShape() -> NovaR1CSShape {
    let m = 1, n = 4, numPublic = 1
    var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
    aBuilder.set(row: 0, col: 2, value: Fr.one)

    var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
    bBuilder.set(row: 0, col: 3, value: Fr.one)

    var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
    cBuilder.set(row: 0, col: 1, value: Fr.one)

    return NovaR1CSShape(numConstraints: m, numVariables: n, numPublicInputs: numPublic,
                         A: aBuilder.build(), B: bBuilder.build(), C: cBuilder.build())
}

/// Build a two-constraint circuit: w1 * w1 = w2, w2 * w1 = y
/// This computes x^3 with z = [1, x, y, w1, w2]
/// numPublic = 2 (x, y), witness = 2 (w1, w2)
private func engineCubingShape() -> NovaR1CSShape {
    let m = 2, n = 5, numPublic = 2
    // Constraint 0: w1 * w1 = w2
    var a0 = SparseMatrixBuilder(rows: m, cols: n)
    a0.set(row: 0, col: 3, value: Fr.one) // w1
    a0.set(row: 1, col: 4, value: Fr.one) // w2

    var b0 = SparseMatrixBuilder(rows: m, cols: n)
    b0.set(row: 0, col: 3, value: Fr.one) // w1
    b0.set(row: 1, col: 3, value: Fr.one) // w1

    var c0 = SparseMatrixBuilder(rows: m, cols: n)
    c0.set(row: 0, col: 4, value: Fr.one) // w2
    c0.set(row: 1, col: 2, value: Fr.one) // y

    return NovaR1CSShape(numConstraints: m, numVariables: n, numPublicInputs: numPublic,
                         A: a0.build(), B: b0.build(), C: c0.build())
}

/// Create a valid squaring instance: x^2 = y
private func engineSquaringPair(_ val: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let x = frFromInt(val)
    let y = frMul(x, x)
    return (NovaR1CSInput(x: [x, y]), NovaR1CSWitness(W: [x]))
}

/// Create a valid multiply instance: a * b = c
private func engineMultiplyPair(_ a: UInt64, _ b: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let fa = frFromInt(a)
    let fb = frFromInt(b)
    let fc = frMul(fa, fb)
    return (NovaR1CSInput(x: [fc]), NovaR1CSWitness(W: [fa, fb]))
}

/// Create a valid cubing instance: x^3 = y
private func engineCubingPair(_ val: UInt64) -> (NovaR1CSInput, NovaR1CSWitness) {
    let x = frFromInt(val)
    let x2 = frMul(x, x)
    let x3 = frMul(x2, x)
    return (NovaR1CSInput(x: [x, x3]), NovaR1CSWitness(W: [x, x2]))
}

// MARK: - Tests

public func runNovaEngineTests() {
    suite("Nova Engine")

    // =========================================================================
    // Test 1: R1CS satisfaction check via shape
    // =========================================================================
    do {
        let shape = engineSquaringShape()
        let (inst, wit) = engineSquaringPair(7)
        let sat = shape.satisfies(instance: inst, witness: wit)
        expect(sat, "7^2=49 should satisfy squaring R1CS")
    }

    // =========================================================================
    // Test 2: R1CS rejects invalid witness
    // =========================================================================
    do {
        let shape = engineSquaringShape()
        let x = frFromInt(4)
        let wrongY = frFromInt(17) // should be 16
        let inst = NovaR1CSInput(x: [x, wrongY])
        let wit = NovaR1CSWitness(W: [x])
        let sat = shape.satisfies(instance: inst, witness: wit)
        expect(!sat, "4^2 != 17, should reject invalid witness")
    }

    // =========================================================================
    // Test 3: Single-step IVC proof (base case only)
    // =========================================================================
    do {
        let shape = engineSquaringShape()
        let prover = NovaIVCProver(shape: shape)
        let verifier = NovaIVCVerifier(shape: shape)

        let (inst, wit) = engineSquaringPair(5)
        let proof = prover.prove(steps: [(inst, wit)])

        expect(proof.stepCount == 1, "Single step should have stepCount=1")
        expect(proof.foldProofs.isEmpty, "Single step has no fold proofs")

        let valid = verifier.verify(proof: proof, pp: prover.pp)
        expect(valid, "Single-step IVC proof should verify")
    }

    // =========================================================================
    // Test 4: Single fold correctness — folded instance satisfies relaxed R1CS
    // =========================================================================
    do {
        let shape = engineSquaringShape()
        let prover = NovaIVCProver(shape: shape)

        let (inst1, wit1) = engineSquaringPair(3)
        let (inst2, wit2) = engineSquaringPair(5)
        let proof = prover.prove(steps: [(inst1, wit1), (inst2, wit2)])

        expect(proof.stepCount == 2, "Two steps should have stepCount=2")
        expect(proof.foldProofs.count == 1, "One fold proof for two steps")

        // Final instance should satisfy relaxed R1CS
        let sat = shape.satisfiesRelaxed(instance: proof.finalInstance,
                                          witness: proof.finalWitness)
        expect(sat, "Folded instance should satisfy relaxed R1CS")
    }

    // =========================================================================
    // Test 5: Multi-step IVC — 6 steps (squaring circuit)
    // =========================================================================
    do {
        let shape = engineSquaringShape()
        let prover = NovaIVCProver(shape: shape)
        let verifier = NovaIVCVerifier(shape: shape)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        for i: UInt64 in 2...7 {
            steps.append(engineSquaringPair(i))
        }

        let proof = prover.prove(steps: steps)

        expect(proof.stepCount == 6, "6 steps should have stepCount=6")
        expect(proof.foldProofs.count == 5, "5 fold proofs for 6 steps")

        // Relaxed R1CS satisfaction
        let sat = shape.satisfiesRelaxed(instance: proof.finalInstance,
                                          witness: proof.finalWitness)
        expect(sat, "6-step IVC should satisfy relaxed R1CS")

        // Full verification (fold checks + decider)
        let valid = verifier.verify(proof: proof, pp: prover.pp)
        expect(valid, "6-step IVC proof should verify")

        // u should not be 1 after folding
        expect(!frEq(proof.finalInstance.u, Fr.one), "After folding, u != 1")
    }

    // =========================================================================
    // Test 6: Multi-step IVC — multiply circuit, 5 steps
    // =========================================================================
    do {
        let shape = engineMultiplyShape()
        let prover = NovaIVCProver(shape: shape)
        let verifier = NovaIVCVerifier(shape: shape)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        steps.append(engineMultiplyPair(3, 5))
        steps.append(engineMultiplyPair(7, 11))
        steps.append(engineMultiplyPair(2, 13))
        steps.append(engineMultiplyPair(4, 9))
        steps.append(engineMultiplyPair(6, 8))

        let proof = prover.prove(steps: steps)
        let valid = verifier.verify(proof: proof, pp: prover.pp)
        expect(valid, "5-step multiply IVC should verify")
    }

    // =========================================================================
    // Test 7: Multi-step IVC — cubing circuit with 2 constraints, 5 steps
    // =========================================================================
    do {
        let shape = engineCubingShape()
        let prover = NovaIVCProver(shape: shape)
        let verifier = NovaIVCVerifier(shape: shape)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        for i: UInt64 in 2...6 {
            steps.append(engineCubingPair(i))
        }

        let proof = prover.prove(steps: steps)

        let sat = shape.satisfiesRelaxed(instance: proof.finalInstance,
                                          witness: proof.finalWitness)
        expect(sat, "Cubing 5-step IVC should satisfy relaxed R1CS")

        let valid = verifier.verify(proof: proof, pp: prover.pp)
        expect(valid, "Cubing 5-step IVC should verify")
    }

    // =========================================================================
    // Test 8: Invalid witness rejection — decider catches bad witness
    // =========================================================================
    do {
        let shape = engineSquaringShape()
        let prover = NovaIVCProver(shape: shape)
        let verifier = NovaIVCVerifier(shape: shape)

        // Create a valid proof first
        let (inst, wit) = engineSquaringPair(4)
        let proof = prover.prove(steps: [(inst, wit)])

        // Tamper with the witness
        var badW = proof.finalWitness.W
        badW[0] = frAdd(badW[0], Fr.one) // corrupt witness
        let badWitness = NovaRelaxedWitness(W: badW, E: proof.finalWitness.E)

        // Decider should reject the tampered witness
        let badValid = verifier.deciderCheck(instance: proof.finalInstance,
                                              witness: badWitness, pp: prover.pp)
        expect(!badValid, "Decider should reject tampered witness")
    }

    // =========================================================================
    // Test 9: Invalid witness rejection — bad witness in fold chain
    // =========================================================================
    do {
        let shape = engineSquaringShape()
        let prover = NovaIVCProver(shape: shape)
        let verifier = NovaIVCVerifier(shape: shape)

        // Build a valid 3-step proof
        let (inst1, wit1) = engineSquaringPair(3)
        let (inst2, wit2) = engineSquaringPair(5)
        let (inst3, wit3) = engineSquaringPair(7)
        let proof = prover.prove(steps: [(inst1, wit1), (inst2, wit2), (inst3, wit3)])

        // The proof should be valid
        let valid = verifier.verify(proof: proof, pp: prover.pp)
        expect(valid, "Valid 3-step proof should verify")

        // Tamper with final witness — decider will catch this
        var badW = proof.finalWitness.W
        badW[0] = frFromInt(999)
        let tamperedProof = NovaIVCProof(
            finalInstance: proof.finalInstance,
            finalWitness: NovaRelaxedWitness(W: badW, E: proof.finalWitness.E),
            foldProofs: proof.foldProofs,
            freshInstances: proof.freshInstances,
            intermediateInstances: proof.intermediateInstances,
            stepCount: proof.stepCount)

        let tamperedValid = verifier.verify(proof: tamperedProof, pp: prover.pp)
        expect(!tamperedValid, "Tampered witness should fail verification")
    }

    // =========================================================================
    // Test 10: Fold preserves satisfiability — each intermediate fold is valid
    // =========================================================================
    do {
        let shape = engineSquaringShape()
        let prover = NovaIVCProver(shape: shape)

        // Use incremental API to check each fold
        let pairs: [(NovaR1CSInput, NovaR1CSWitness)] = [
            engineSquaringPair(2),
            engineSquaringPair(3),
            engineSquaringPair(4),
            engineSquaringPair(5),
            engineSquaringPair(6),
            engineSquaringPair(7),
        ]

        prover.initialize(instance: pairs[0].0, witness: pairs[0].1)

        // Check base case satisfies relaxed R1CS
        let baseSat = shape.satisfiesRelaxed(instance: prover.runningInstance!,
                                              witness: prover.runningWitness!)
        expect(baseSat, "Base case should satisfy relaxed R1CS")

        // Fold each step and check satisfaction after each fold
        for i in 1..<pairs.count {
            prover.foldStep(instance: pairs[i].0, witness: pairs[i].1)
            let sat = shape.satisfiesRelaxed(instance: prover.runningInstance!,
                                              witness: prover.runningWitness!)
            expect(sat, "After fold \(i), accumulated instance should satisfy relaxed R1CS")
        }

        expect(prover.stepCount == 6, "Should have 6 steps")
    }

    // =========================================================================
    // Test 11: Fold-only verification (no decider)
    // =========================================================================
    do {
        let shape = engineSquaringShape()
        let prover = NovaIVCProver(shape: shape)
        let verifier = NovaIVCVerifier(shape: shape)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        for i: UInt64 in 3...8 {
            steps.append(engineSquaringPair(i))
        }

        let proof = prover.prove(steps: steps)

        // Fold-only verification should pass
        let foldOk = verifier.verifyFoldsOnly(proof: proof)
        expect(foldOk, "Fold-only verification should pass for valid proof")
    }

    // =========================================================================
    // Test 12: Tampered fold proof rejected by fold-only verifier
    // =========================================================================
    do {
        let shape = engineSquaringShape()
        let prover = NovaIVCProver(shape: shape)
        let verifier = NovaIVCVerifier(shape: shape)

        let (inst1, wit1) = engineSquaringPair(4)
        let (inst2, wit2) = engineSquaringPair(6)
        let (inst3, wit3) = engineSquaringPair(8)
        let proof = prover.prove(steps: [(inst1, wit1), (inst2, wit2), (inst3, wit3)])

        // Tamper with a fold proof: change the commitment to T
        let g = pointFromAffine(bn254G1Generator())
        let fakeCommitT = cPointScalarMul(g, frFromInt(12345))
        var badProofs = proof.foldProofs
        badProofs[0] = NovaFoldProof(commitT: fakeCommitT)

        let tamperedProof = NovaIVCProof(
            finalInstance: proof.finalInstance,
            finalWitness: proof.finalWitness,
            foldProofs: badProofs,
            freshInstances: proof.freshInstances,
            intermediateInstances: proof.intermediateInstances,
            stepCount: proof.stepCount)

        // Fold verification should fail because the re-derived r won't match
        let foldOk = verifier.verifyFoldsOnly(proof: tamperedProof)
        expect(!foldOk, "Tampered fold proof should be rejected")
    }

    // =========================================================================
    // Test 13: Step circuit interface — closure-based step
    // =========================================================================
    do {
        let shape = engineSquaringShape()
        let prover = NovaIVCProver(shape: shape)

        // Define a squaring step circuit via closure
        let circuit = NovaClosureStep { stepIndex, stateIn -> (publicInput: [Fr], witness: [Fr]) in
            // stateIn is the value to square
            let val = frFromInt(UInt64(stepIndex + 2))
            let sq = frMul(val, val)
            return (publicInput: [val, sq], witness: [val])
        }

        let proof = prover.prove(circuit: circuit, numSteps: 5,
                                  initialState: [frFromInt(2)])

        expect(proof.stepCount == 5, "Circuit-based IVC should have 5 steps")

        let sat = shape.satisfiesRelaxed(instance: proof.finalInstance,
                                          witness: proof.finalWitness)
        expect(sat, "Circuit-based IVC should satisfy relaxed R1CS")
    }

    // =========================================================================
    // Test 14: Non-zero error vector after folding
    // =========================================================================
    do {
        let shape = engineSquaringShape()
        let prover = NovaIVCProver(shape: shape)

        let (inst1, wit1) = engineSquaringPair(3)
        let (inst2, wit2) = engineSquaringPair(7)
        let proof = prover.prove(steps: [(inst1, wit1), (inst2, wit2)])

        // After folding, E should be non-zero
        let hasNonZeroE = proof.finalWitness.E.contains { !$0.isZero }
        expect(hasNonZeroE, "Error vector should be non-zero after folding")

        // But relaxed R1CS should still hold
        let sat = shape.satisfiesRelaxed(instance: proof.finalInstance,
                                          witness: proof.finalWitness)
        expect(sat, "Relaxed R1CS holds despite non-zero error")
    }

    // =========================================================================
    // Test 15: Large IVC chain — 10 steps
    // =========================================================================
    do {
        let shape = engineSquaringShape()
        let prover = NovaIVCProver(shape: shape)
        let verifier = NovaIVCVerifier(shape: shape)

        var steps = [(instance: NovaR1CSInput, witness: NovaR1CSWitness)]()
        for i: UInt64 in 1...10 {
            steps.append(engineSquaringPair(i))
        }

        let proof = prover.prove(steps: steps)

        expect(proof.stepCount == 10, "10-step chain should have stepCount=10")
        expect(proof.foldProofs.count == 9, "9 fold proofs for 10 steps")

        let valid = verifier.verify(proof: proof, pp: prover.pp)
        expect(valid, "10-step IVC proof should verify")
    }
}
