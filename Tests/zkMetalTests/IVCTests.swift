import zkMetal

// MARK: - Test Helpers

/// Build a simple R1CS circuit: x * x = y (squaring)
/// Variables: z = [1, x, y, w] where w = x (witness copy)
/// Constraint: w * w = y => A=[0,0,0,1], B=[0,0,0,1], C=[0,0,1,0]
/// Public: x, y (numPublic=2), Witness: w (1 element)
private func makeSquaringCircuit() -> (A: SparseMatrix, B: SparseMatrix, C: SparseMatrix, numPublic: Int) {
    // z = [1, x, y, w]  (n=4, m=1, numPublic=2)
    // Constraint: w * w = y
    let m = 1, n = 4
    var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
    aBuilder.set(row: 0, col: 3, value: Fr.one) // w

    var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
    bBuilder.set(row: 0, col: 3, value: Fr.one) // w

    var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
    cBuilder.set(row: 0, col: 2, value: Fr.one) // y

    return (aBuilder.build(), bBuilder.build(), cBuilder.build(), 2)
}

/// Build a multiply circuit: a * b = c
/// Variables: z = [1, c, a, b] (numPublic=1 for output c, witness: a, b)
/// Constraint: a * b = c
private func makeMultiplyCircuit() -> (A: SparseMatrix, B: SparseMatrix, C: SparseMatrix, numPublic: Int) {
    let m = 1, n = 4
    var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
    aBuilder.set(row: 0, col: 2, value: Fr.one) // a

    var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
    bBuilder.set(row: 0, col: 3, value: Fr.one) // b

    var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
    cBuilder.set(row: 0, col: 1, value: Fr.one) // c

    return (aBuilder.build(), bBuilder.build(), cBuilder.build(), 1)
}

/// Build an addition circuit: a + b = c using R1CS trick: (a+b) * 1 = c
/// Variables: z = [1, c, a, b] (numPublic=1, witness: a, b)
private func makeAddCircuit() -> (A: SparseMatrix, B: SparseMatrix, C: SparseMatrix, numPublic: Int) {
    let m = 1, n = 4
    // A: row0 = [0, 0, 1, 1]  => a + b
    var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
    aBuilder.set(row: 0, col: 2, value: Fr.one) // a
    aBuilder.set(row: 0, col: 3, value: Fr.one) // b

    // B: row0 = [1, 0, 0, 0]  => 1
    var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
    bBuilder.set(row: 0, col: 0, value: Fr.one)

    // C: row0 = [0, 1, 0, 0]  => c
    var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
    cBuilder.set(row: 0, col: 1, value: Fr.one)

    return (aBuilder.build(), bBuilder.build(), cBuilder.build(), 1)
}

// MARK: - Nova Tests

func runIVCTests() {
    suite("Nova IVC")

    // -- Test: R1CS satisfaction for the squaring circuit --
    do {
        let (A, B, C, numPub) = makeSquaringCircuit()
        let step = NovaStep(A: A, B: B, C: C, numPublic: numPub)
        // x=3, y=9, w=3: w*w = 9 = y
        let x = frFromInt(3)
        let y = frFromInt(9)
        let w = frFromInt(3)
        expect(step.isSatisfied(publicInput: [x, y], witness: [w]),
               "Squaring circuit satisfied for x=3")
        // Bad witness: w=4, 4*4=16 != 9
        expect(!step.isSatisfied(publicInput: [x, y], witness: [frFromInt(4)]),
               "Squaring circuit rejects bad witness")
    }

    // -- Test: Nova single step (initialize) --
    do {
        let (A, B, C, numPub) = makeSquaringCircuit()
        let step = NovaStep(A: A, B: B, C: C, numPublic: numPub)
        let prover = NovaProver(step: step)

        let x = frFromInt(3)
        let y = frFromInt(9)
        let w = frFromInt(3)
        let inst = prover.initialize(publicInput: [x, y], witness: [w])

        // After initialize: u=1 (fresh), E=identity, step count=1
        expect(frEq(inst.u, Fr.one), "Initial u = 1")
        expect(pointIsIdentity(inst.commitE), "Initial E = identity")
        expectEqual(prover.stepCount, 1, "Step count after init")

        // Decider should accept
        let decider = NovaDecider(step: step)
        expect(decider.decide(instance: prover.runningInstance!,
                              witness: prover.runningWitness!),
               "Decider accepts single step")
    }

    // -- Test: Nova multi-step folding (4 steps of squaring) --
    do {
        let (A, B, C, numPub) = makeSquaringCircuit()
        let step = NovaStep(A: A, B: B, C: C, numPublic: numPub)
        let prover = NovaProver(step: step)

        // Step 0: x=2, y=4, w=2
        prover.initialize(publicInput: [frFromInt(2), frFromInt(4)], witness: [frFromInt(2)])

        // Steps 1-3: more squaring instances
        let values: [(UInt64, UInt64)] = [(3, 9), (5, 25), (7, 49)]
        var proofs = [NovaFoldingProof]()
        for (x, y) in values {
            let (_, proof) = prover.prove(publicInput: [frFromInt(x), frFromInt(y)],
                                          witness: [frFromInt(x)])
            proofs.append(proof)
        }

        expectEqual(prover.stepCount, 4, "Step count after 4 steps")

        // After folding, u should no longer be 1
        expect(!frEq(prover.runningInstance!.u, Fr.one), "Folded u != 1")

        // Commitment to E should no longer be identity
        expect(!pointIsIdentity(prover.runningInstance!.commitE), "Folded E != identity")

        // Decider should accept the accumulated instance
        let decider = NovaDecider(step: step)
        expect(decider.decide(instance: prover.runningInstance!,
                              witness: prover.runningWitness!),
               "Decider accepts 4-step folded instance")
    }

    // -- Test: Nova 8-step chain via ivcChain convenience --
    do {
        let (A, B, C, numPub) = makeSquaringCircuit()
        let step = NovaStep(A: A, B: B, C: C, numPublic: numPub)
        let prover = NovaProver(step: step)

        var steps = [(publicInput: [Fr], witness: [Fr])]()
        for i: UInt64 in 1...8 {
            steps.append((publicInput: [frFromInt(i), frFromInt(i * i)], witness: [frFromInt(i)]))
        }

        let (finalInst, count) = prover.ivcChain(steps: steps)
        expectEqual(count, 8, "ivcChain step count")

        let decider = NovaDecider(step: step)
        expect(decider.decide(instance: finalInst, witness: prover.runningWitness!),
               "Decider accepts 8-step ivcChain")
    }

    // -- Test: Nova cross-term T computation --
    // Verify that after a single fold, the relaxed R1CS relation holds:
    //   Az . Bz = u*(Cz) + E
    do {
        let (A, B, C, numPub) = makeSquaringCircuit()
        let step = NovaStep(A: A, B: B, C: C, numPublic: numPub)
        let prover = NovaProver(step: step)

        prover.initialize(publicInput: [frFromInt(3), frFromInt(9)], witness: [frFromInt(3)])
        let (foldedInst, _) = prover.prove(publicInput: [frFromInt(5), frFromInt(25)],
                                           witness: [frFromInt(5)])
        let foldedWit = prover.runningWitness!

        // Manually check relaxed R1CS: Az . Bz = u*(Cz) + E
        let z = step.buildRelaxedZ(u: foldedInst.u, publicInput: foldedInst.publicInput,
                                    witness: foldedWit.W)
        let Az = step.r1cs.matrices[0].mulVec(z)
        let Bz = step.r1cs.matrices[1].mulVec(z)
        let Cz = step.r1cs.matrices[2].mulVec(z)

        for i in 0..<step.r1cs.m {
            let lhs = frMul(Az[i], Bz[i])
            let rhs = frAdd(frMul(foldedInst.u, Cz[i]), foldedWit.E[i])
            expect(frEq(lhs, rhs), "Cross-term T: relaxed R1CS row \(i) holds after fold")
        }
    }

    // -- Test: Nova verifier checks fold correctness --
    do {
        let (A, B, C, numPub) = makeSquaringCircuit()
        let step = NovaStep(A: A, B: B, C: C, numPublic: numPub)
        let prover = NovaProver(step: step)
        let verifier = NovaVerifier(step: step)

        let initInst = prover.initialize(publicInput: [frFromInt(4), frFromInt(16)],
                                         witness: [frFromInt(4)])

        let newPub: [Fr] = [frFromInt(6), frFromInt(36)]
        let newWit: [Fr] = [frFromInt(6)]
        let newCommitW = step.pp.commit(witness: newWit)
        let (foldedInst, proof) = prover.prove(publicInput: newPub, witness: newWit)

        let ok = verifier.verifyFold(running: initInst,
                                     newCommitW: newCommitW,
                                     newPublicInput: newPub,
                                     folded: foldedInst,
                                     proof: proof)
        expect(ok, "Verifier accepts valid fold")
    }

    // -- Test: Nova rejects bad witness via decider --
    do {
        let (A, B, C, numPub) = makeSquaringCircuit()
        let step = NovaStep(A: A, B: B, C: C, numPublic: numPub)
        let prover = NovaProver(step: step)

        // Initialize with a correct step
        prover.initialize(publicInput: [frFromInt(3), frFromInt(9)], witness: [frFromInt(3)])

        // Fold a WRONG witness: claim 4*4=9 (wrong, should be 16)
        let (foldedInst, _) = prover.prove(publicInput: [frFromInt(4), frFromInt(9)],
                                           witness: [frFromInt(4)])

        // The decider should reject because the folded relaxed R1CS won't hold
        let decider = NovaDecider(step: step)
        let ok = decider.decide(instance: foldedInst, witness: prover.runningWitness!)
        expect(!ok, "Decider rejects folded instance with invalid witness")
    }

    // -- Test: Nova multiply circuit --
    do {
        let (A, B, C, numPub) = makeMultiplyCircuit()
        let step = NovaStep(A: A, B: B, C: C, numPublic: numPub)
        let prover = NovaProver(step: step)

        // a=3, b=7 => c=21
        prover.initialize(publicInput: [frFromInt(21)], witness: [frFromInt(3), frFromInt(7)])
        // a=5, b=4 => c=20
        let (_, _) = prover.prove(publicInput: [frFromInt(20)], witness: [frFromInt(5), frFromInt(4)])
        // a=6, b=6 => c=36
        let (_, _) = prover.prove(publicInput: [frFromInt(36)], witness: [frFromInt(6), frFromInt(6)])

        let decider = NovaDecider(step: step)
        expect(decider.decide(instance: prover.runningInstance!, witness: prover.runningWitness!),
               "Multiply circuit 3-step fold accepted by decider")
    }

    suite("SuperNova IVC")

    // -- Test: SuperNova with 2 circuits (multiply and add), round-robin --
    do {
        let (mA, mB, mC, mPub) = makeMultiplyCircuit()
        let (aA, aB, aC, aPub) = makeAddCircuit()
        let mulStep = NovaStep(A: mA, B: mB, C: mC, numPublic: mPub)
        let addStep = NovaStep(A: aA, B: aB, C: aC, numPublic: aPub)

        // Round-robin selector: step 0 -> mul, step 1 -> add, step 2 -> mul, ...
        let selector = CircuitSelector.roundRobin(count: 2)
        let prover = SuperNovaProver(steps: [mulStep, addStep], selector: selector)

        // Step 0 (mul): a=3, b=4 => c=12
        prover.initialize(publicInput: [frFromInt(12)],
                          witness: [frFromInt(3), frFromInt(4)],
                          initialState: [frFromInt(12)])

        expectEqual(prover.state!.stepCount, 1, "SuperNova step count after init")
        expectEqual(prover.state!.lastCircuitIndex, 0, "SuperNova init used circuit 0 (mul)")

        // Step 1 (add): a=5, b=7 => c=12
        let (state1, proof1) = prover.prove(publicInput: [frFromInt(12)],
                                            witness: [frFromInt(5), frFromInt(7)],
                                            newState: [frFromInt(12)])
        expectEqual(proof1.circuitIndex, 1, "Step 1 used circuit 1 (add)")
        expectEqual(state1.stepCount, 2, "SuperNova step count after step 1")

        // Step 2 (mul): a=6, b=3 => c=18
        let (state2, proof2) = prover.prove(publicInput: [frFromInt(18)],
                                            witness: [frFromInt(6), frFromInt(3)],
                                            newState: [frFromInt(18)])
        expectEqual(proof2.circuitIndex, 0, "Step 2 used circuit 0 (mul)")
        expectEqual(state2.stepCount, 3, "SuperNova step count after step 2")

        // Step 3 (add): a=10, b=8 => c=18
        let (_, proof3) = prover.prove(publicInput: [frFromInt(18)],
                                       witness: [frFromInt(10), frFromInt(8)],
                                       newState: [frFromInt(18)])
        expectEqual(proof3.circuitIndex, 1, "Step 3 used circuit 1 (add)")

        // Decider should accept all accumulated instances
        expect(prover.decide(), "SuperNova decider accepts 2-circuit 4-step chain")
    }

    // -- Test: SuperNova with uniform circuit (should behave like Nova) --
    do {
        let (A, B, C, numPub) = makeSquaringCircuit()
        let step = NovaStep(A: A, B: B, C: C, numPublic: numPub)

        // Use constant selector (always circuit 0) with 1 circuit type
        let selector = CircuitSelector.constant(index: 0, count: 1)
        let prover = SuperNovaProver(steps: [step], selector: selector)

        prover.initialize(publicInput: [frFromInt(2), frFromInt(4)],
                          witness: [frFromInt(2)],
                          initialState: [frFromInt(4)])

        for i: UInt64 in 3...6 {
            let (_, _) = prover.prove(publicInput: [frFromInt(i), frFromInt(i * i)],
                                      witness: [frFromInt(i)],
                                      newState: [frFromInt(i * i)])
        }

        expectEqual(prover.state!.stepCount, 5, "Uniform SuperNova step count")

        // All steps used circuit 0
        expectEqual(prover.state!.lastCircuitIndex, 0, "Uniform SuperNova always circuit 0")

        // Decider should accept
        expect(prover.decide(), "Uniform SuperNova decider accepts (reduces to Nova)")
    }

    // -- Test: SuperNova non-folded circuits remain unchanged across steps --
    do {
        let (mA, mB, mC, mPub) = makeMultiplyCircuit()
        let (aA, aB, aC, aPub) = makeAddCircuit()
        let mulStep = NovaStep(A: mA, B: mB, C: mC, numPublic: mPub)
        let addStep = NovaStep(A: aA, B: aB, C: aC, numPublic: aPub)

        let selector = CircuitSelector.roundRobin(count: 2)
        let prover = SuperNovaProver(steps: [mulStep, addStep], selector: selector)

        // Step 0 (mul): a=2, b=5 => c=10
        prover.initialize(publicInput: [frFromInt(10)],
                          witness: [frFromInt(2), frFromInt(5)],
                          initialState: [frFromInt(10)])
        let prevState = prover.state!

        // Step 1 (add): a=3, b=4 => c=7  -- only circuit 1 is folded
        let (nextState, proof) = prover.prove(publicInput: [frFromInt(7)],
                                              witness: [frFromInt(3), frFromInt(4)],
                                              newState: [frFromInt(7)])

        expectEqual(proof.circuitIndex, 1, "Step 1 selected add circuit")

        // Circuit 0 (mul) should be completely unchanged
        expect(pointEqual(prevState.runningInstances[0].commitW,
                          nextState.runningInstances[0].commitW),
               "Non-folded circuit commitW unchanged")
        expect(pointEqual(prevState.runningInstances[0].commitE,
                          nextState.runningInstances[0].commitE),
               "Non-folded circuit commitE unchanged")
        expect(frEq(prevState.runningInstances[0].u,
                    nextState.runningInstances[0].u),
               "Non-folded circuit u unchanged")

        // Circuit 1 (add) SHOULD have changed
        expect(!frEq(prevState.runningInstances[1].u,
                     nextState.runningInstances[1].u),
               "Folded circuit u changed after fold")
    }

    // -- Test: SuperNova rejects bad witness --
    do {
        let (A, B, C, numPub) = makeMultiplyCircuit()
        let step = NovaStep(A: A, B: B, C: C, numPublic: numPub)

        let selector = CircuitSelector.constant(index: 0, count: 1)
        let prover = SuperNovaProver(steps: [step], selector: selector)

        // Valid step 0: 3*7=21
        prover.initialize(publicInput: [frFromInt(21)],
                          witness: [frFromInt(3), frFromInt(7)],
                          initialState: [frFromInt(21)])

        // Invalid step 1: claim 2*5=99 (wrong, should be 10)
        let (_, _) = prover.prove(publicInput: [frFromInt(99)],
                                  witness: [frFromInt(2), frFromInt(5)],
                                  newState: [frFromInt(99)])

        // Decider should reject
        expect(!prover.decide(), "SuperNova decider rejects chain with invalid witness")
    }

    // -- Test: SuperNova ivcChain convenience --
    do {
        let (A, B, C, numPub) = makeMultiplyCircuit()
        let step = NovaStep(A: A, B: B, C: C, numPublic: numPub)

        let selector = CircuitSelector.constant(index: 0, count: 1)
        let prover = SuperNovaProver(steps: [step], selector: selector)

        let chainSteps: [(publicInput: [Fr], witness: [Fr], newState: [Fr])] = [
            ([frFromInt(6)],  [frFromInt(2), frFromInt(3)], [frFromInt(6)]),
            ([frFromInt(20)], [frFromInt(4), frFromInt(5)], [frFromInt(20)]),
            ([frFromInt(42)], [frFromInt(6), frFromInt(7)], [frFromInt(42)]),
            ([frFromInt(72)], [frFromInt(8), frFromInt(9)], [frFromInt(72)]),
        ]

        let (finalState, count) = prover.ivcChain(initialState: [frFromInt(0)], steps: chainSteps)
        expectEqual(count, 4, "SuperNova ivcChain step count")
        expectEqual(finalState.stepCount, 4, "SuperNova ivcChain final state step count")
        expect(prover.decide(), "SuperNova ivcChain decider accepts")
    }

    // -- Test: CircuitSelector roundRobin --
    do {
        let sel = CircuitSelector.roundRobin(count: 3)
        expectEqual(sel.select(0, []), 0, "roundRobin step 0")
        expectEqual(sel.select(1, []), 1, "roundRobin step 1")
        expectEqual(sel.select(2, []), 2, "roundRobin step 2")
        expectEqual(sel.select(3, []), 0, "roundRobin step 3 wraps")
        expectEqual(sel.select(7, []), 1, "roundRobin step 7")
    }

    // -- Test: CircuitSelector constant --
    do {
        let sel = CircuitSelector.constant(index: 2, count: 4)
        expectEqual(sel.select(0, []), 2, "constant selector always 2")
        expectEqual(sel.select(5, []), 2, "constant selector still 2")
    }
}
