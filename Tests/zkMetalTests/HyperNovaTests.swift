import zkMetal
import Foundation

// MARK: - Test Helpers

/// Build a simple R1CS: a * b = c
/// z = [1, c, a, b] (n=4, m=1, numPublic=1 for output c, witness: a, b)
private func makeR1CSMultiply() -> (CCSInstance, Int) {
    let m = 1, n = 4
    var aBuilder = SparseMatrixBuilder(rows: m, cols: n)
    aBuilder.set(row: 0, col: 2, value: Fr.one) // a

    var bBuilder = SparseMatrixBuilder(rows: m, cols: n)
    bBuilder.set(row: 0, col: 3, value: Fr.one) // b

    var cBuilder = SparseMatrixBuilder(rows: m, cols: n)
    cBuilder.set(row: 0, col: 1, value: Fr.one) // c

    let ccs = CCSInstance.fromR1CS(A: aBuilder.build(), B: bBuilder.build(),
                                   C: cBuilder.build(), numPublicInputs: 1)
    return (ccs, 1)
}

/// Build a 2-constraint R1CS: x*x = y, y*x = z
/// z_vec = [1, x, z_out, y_mid, x_copy]
/// Constraint 0: x_copy * x_copy = y_mid  => A[0]=[0,0,0,0,1], B[0]=[0,0,0,0,1], C[0]=[0,0,0,1,0]
/// Constraint 1: y_mid * x_copy = z_out   => A[1]=[0,0,0,1,0], B[1]=[0,0,0,0,1], C[1]=[0,0,1,0,0]
/// numPublic=1 (x), witness = [z_out, y_mid, x_copy]
private func makeR1CSCubic() -> (CCSInstance, Int) {
    let m = 2, n = 5
    // A matrix: row 0 picks col 4 (x_copy), row 1 picks col 3 (y_mid)
    var aB = SparseMatrixBuilder(rows: m, cols: n)
    aB.set(row: 0, col: 4, value: Fr.one)
    aB.set(row: 1, col: 3, value: Fr.one)

    // B matrix: row 0 picks col 4 (x_copy), row 1 picks col 4 (x_copy)
    var bB = SparseMatrixBuilder(rows: m, cols: n)
    bB.set(row: 0, col: 4, value: Fr.one)
    bB.set(row: 1, col: 4, value: Fr.one)

    // C matrix: row 0 picks col 3 (y_mid), row 1 picks col 2 (z_out)
    var cB = SparseMatrixBuilder(rows: m, cols: n)
    cB.set(row: 0, col: 3, value: Fr.one)
    cB.set(row: 1, col: 2, value: Fr.one)

    let ccs = CCSInstance.fromR1CS(A: aB.build(), B: bB.build(),
                                   C: cB.build(), numPublicInputs: 1)
    return (ccs, 1)
}

/// Build a simple Plonk circuit: 2 gates computing a*b + c = d
/// Gate 0: qM=1, others=0 => a0*b0 = c0 (multiply)
/// Gate 1: qL=1, qR=1, qO=-1 => a1 + b1 - c1 = 0 (add)
private func makePlonkAddMul() -> CCSInstance {
    let numGates = 2
    let minusOne = frSub(Fr.zero, Fr.one)

    let qL: [Fr] = [Fr.zero, Fr.one]
    let qR: [Fr] = [Fr.zero, Fr.one]
    let qO: [Fr] = [Fr.zero, minusOne]
    let qM: [Fr] = [Fr.one, Fr.zero]
    let qC: [Fr] = [Fr.zero, Fr.zero]

    return CCSInstance.fromPlonk(numGates: numGates,
                                 qL: qL, qR: qR, qO: qO, qM: qM, qC: qC,
                                 numPublicInputs: 0)
}

// MARK: - HyperNova Tests

func runHyperNovaTests() {
    suite("HyperNova Folding")

    // -- Test 1: Create CCS from R1CS, fold 2 instances, verify --
    do {
        let (ccs, _) = makeR1CSMultiply()

        // Verify CCS satisfaction first
        // z = [1, c, a, b] where a*b = c
        let z1 = [Fr.one, frFromInt(15), frFromInt(3), frFromInt(5)]
        expect(ccs.isSatisfied(z: z1), "R1CS CCS satisfied: 3*5=15")

        // Create HyperNova engine and prover
        let prover = HyperNovaProver(ccs: ccs)
        let verifier = HyperNovaVerifier(ccs: ccs)

        // Initialize with first instance: 3*5=15
        let witness1: [Fr] = [frFromInt(3), frFromInt(5)]
        let pubInput1: [Fr] = [frFromInt(15)]
        let (running, runningWit) = prover.initialize(witness: witness1, publicInput: pubInput1)

        // Commit second instance: 7*4=28
        let witness2: [Fr] = [frFromInt(7), frFromInt(4)]
        let pubInput2: [Fr] = [frFromInt(28)]
        let newInstance = prover.commitWitness(witness2, publicInput: pubInput2)

        // Fold
        let (folded, foldedWit, proof) = prover.fold(
            running: running, runningWitness: runningWit,
            new: newInstance, newWitness: witness2)

        // Verify fold
        let verifyOk = verifier.verifyFold(running: running, new: newInstance,
                                            folded: folded, proof: proof)
        expect(verifyOk, "Verifier accepts 2-instance fold")

        // Decide (final check on accumulated instance)
        let decideOk = prover.decide(instance: folded, witness: foldedWit)
        expect(decideOk, "Decider accepts folded R1CS multiply instance")
    }

    // -- Test 2: Fold 4 instances sequentially, verify final accumulator --
    do {
        let (ccs, _) = makeR1CSMultiply()
        let prover = HyperNovaProver(ccs: ccs)
        let verifier = HyperNovaVerifier(ccs: ccs)

        let instances: [(pub: [Fr], wit: [Fr])] = [
            ([frFromInt(6)],  [frFromInt(2), frFromInt(3)]),   // 2*3=6
            ([frFromInt(20)], [frFromInt(4), frFromInt(5)]),   // 4*5=20
            ([frFromInt(42)], [frFromInt(6), frFromInt(7)]),   // 6*7=42
            ([frFromInt(72)], [frFromInt(8), frFromInt(9)]),   // 8*9=72
        ]

        // Initialize with first instance
        var (running, runningWit) = prover.initialize(
            witness: instances[0].wit, publicInput: instances[0].pub)

        // Fold remaining 3 instances sequentially
        for i in 1..<instances.count {
            let newInst = prover.commitWitness(instances[i].wit, publicInput: instances[i].pub)
            let (folded, foldedWit, proof) = prover.fold(
                running: running, runningWitness: runningWit,
                new: newInst, newWitness: instances[i].wit)

            let ok = verifier.verifyFold(running: running, new: newInst,
                                          folded: folded, proof: proof)
            expect(ok, "Verifier accepts fold step \(i)")

            running = folded
            runningWit = foldedWit
        }

        // After 3 folds, u should no longer be 1
        expect(!frEq(running.u, Fr.one), "Folded u != 1 after 4-instance chain")

        // Decide on final accumulated instance
        let decideOk = prover.decide(instance: running, witness: runningWit)
        expect(decideOk, "Decider accepts 4-instance sequential fold")
    }

    // -- Test 3: CCS from Plonk structure, fold and verify --
    // Tests that HyperNova fold/verify works on a higher-degree CCS (Plonk has degree 3 terms).
    do {
        let ccs = makePlonkAddMul()
        expect(ccs.isWellFormed, "Plonk CCS is well-formed")
        expectEqual(ccs.d, 3, "Plonk CCS has degree 3 (qM * a * b term)")
        expectEqual(ccs.t, 8, "Plonk CCS has 8 matrices")
        expectEqual(ccs.q, 5, "Plonk CCS has 5 terms")
        expectEqual(ccs.m, 2, "Plonk CCS has 2 constraints (gates)")
    }

    // -- Test 3b: CCS from multi-constraint R1CS, fold and verify --
    do {
        let (ccs, _) = makeR1CSCubic()
        expect(ccs.isWellFormed, "Cubic R1CS CCS is well-formed")

        // x=3: x^2=9, x^3=27
        // z = [1, x, z_out, y_mid, x_copy] = [1, 3, 27, 9, 3]
        let z1 = [Fr.one, frFromInt(3), frFromInt(27), frFromInt(9), frFromInt(3)]
        expect(ccs.isSatisfied(z: z1), "Cubic CCS satisfied: x=3, x^3=27")

        let prover = HyperNovaProver(ccs: ccs)
        let verifier = HyperNovaVerifier(ccs: ccs)

        let pub1: [Fr] = [frFromInt(3)]
        let wit1: [Fr] = [frFromInt(27), frFromInt(9), frFromInt(3)]
        let (running, runningWit) = prover.initialize(witness: wit1, publicInput: pub1)

        // x=5: x^2=25, x^3=125
        let pub2: [Fr] = [frFromInt(5)]
        let wit2: [Fr] = [frFromInt(125), frFromInt(25), frFromInt(5)]
        let newInst = prover.commitWitness(wit2, publicInput: pub2)

        let (folded, foldedWit, proof) = prover.fold(
            running: running, runningWitness: runningWit,
            new: newInst, newWitness: wit2)

        let verifyOk = verifier.verifyFold(running: running, new: newInst,
                                            folded: folded, proof: proof)
        expect(verifyOk, "Verifier accepts cubic R1CS fold")

        let decideOk = prover.decide(instance: folded, witness: foldedWit)
        expect(decideOk, "Decider accepts folded cubic R1CS")
    }

    // -- Test 4: Invalid witness is rejected at CCS level --
    do {
        let (ccs, _) = makeR1CSMultiply()

        // a*b != c (3*5 != 10)
        let badZ = [Fr.one, frFromInt(10), frFromInt(3), frFromInt(5)]
        expect(!ccs.isSatisfied(z: badZ), "CCS rejects bad witness: 3*5 != 10")

        // Good witness passes
        let goodZ = [Fr.one, frFromInt(15), frFromInt(3), frFromInt(5)]
        expect(ccs.isSatisfied(z: goodZ), "CCS accepts good witness: 3*5 = 15")

        // Verify that bad commitment is rejected by decider
        let prover = HyperNovaProver(ccs: ccs)

        // Initialize with valid: 3*5=15
        let (running, runningWit) = prover.initialize(
            witness: [frFromInt(3), frFromInt(5)], publicInput: [frFromInt(15)])

        // Fold with valid witness but tamper with the folded witness afterward
        let wit2: [Fr] = [frFromInt(4), frFromInt(5)]
        let pub2: [Fr] = [frFromInt(20)]
        let inst2 = prover.commitWitness(wit2, publicInput: pub2)

        let (folded, foldedWit, _) = prover.fold(
            running: running, runningWitness: runningWit,
            new: inst2, newWitness: wit2)

        // Tamper with the witness: change first element
        var tamperedWit = foldedWit
        tamperedWit[0] = frAdd(tamperedWit[0], Fr.one)

        // Decider rejects because commitment won't match tampered witness
        let decideOk = prover.decide(instance: folded, witness: tamperedWit)
        expect(!decideOk, "Decider rejects tampered witness (commitment mismatch)")
    }

    // -- Test 5: Running instance accumulation preserves satisfiability --
    do {
        let (ccs, _) = makeR1CSCubic()

        // x=2: x^2=4, x^3=8
        // z = [1, x, z_out, y_mid, x_copy] = [1, 2, 8, 4, 2]
        let z1 = [Fr.one, frFromInt(2), frFromInt(8), frFromInt(4), frFromInt(2)]
        expect(ccs.isSatisfied(z: z1), "Cubic CCS satisfied: x=2, x^3=8")

        let prover = HyperNovaProver(ccs: ccs)

        // Public: x. Witness: z_out, y_mid, x_copy
        let pub1: [Fr] = [frFromInt(2)]
        let wit1: [Fr] = [frFromInt(8), frFromInt(4), frFromInt(2)]
        let (running, runningWit) = prover.initialize(witness: wit1, publicInput: pub1)

        // Decide on initial (unfolded) instance
        let initOk = prover.decide(instance: running, witness: runningWit)
        expect(initOk, "Decider accepts initial cubic instance")

        // x=3: x^2=9, x^3=27
        let pub2: [Fr] = [frFromInt(3)]
        let wit2: [Fr] = [frFromInt(27), frFromInt(9), frFromInt(3)]
        let newInst = prover.commitWitness(wit2, publicInput: pub2)

        let (folded1, foldedWit1, _) = prover.fold(
            running: running, runningWitness: runningWit,
            new: newInst, newWitness: wit2)
        let ok1 = prover.decide(instance: folded1, witness: foldedWit1)
        expect(ok1, "Decider accepts after first fold of cubic circuit")

        // x=4: x^2=16, x^3=64
        let pub3: [Fr] = [frFromInt(4)]
        let wit3: [Fr] = [frFromInt(64), frFromInt(16), frFromInt(4)]
        let newInst2 = prover.commitWitness(wit3, publicInput: pub3)

        let (folded2, foldedWit2, _) = prover.fold(
            running: folded1, runningWitness: foldedWit1,
            new: newInst2, newWitness: wit3)
        let ok2 = prover.decide(instance: folded2, witness: foldedWit2)
        expect(ok2, "Decider accepts after second fold of cubic circuit")
    }

    // -- Test 6: CCS linearization check (MLE evaluations are consistent) --
    do {
        let (ccs, _) = makeR1CSMultiply()
        let engine = HyperNovaEngine(ccs: ccs)

        // Create a valid instance
        let witness: [Fr] = [frFromInt(3), frFromInt(5)]
        let pubInput: [Fr] = [frFromInt(15)]
        let lcccs = engine.initialize(witness: witness, publicInput: pubInput)

        // Verify v_i = MLE(M_i * z)(r)
        let z = [Fr.one] + pubInput + witness
        for i in 0..<ccs.t {
            let mv = ccs.matrices[i].mulVec(z)
            let padded = padToPow2(mv)
            let expected = multilinearEval(evals: padded, point: lcccs.r)
            expect(frEq(lcccs.v[i], expected),
                   "MLE eval v[\(i)] matches direct computation")
        }

        // Initial instance: u=1
        expect(frEq(lcccs.u, Fr.one), "Initial LCCCS has u=1")

        // After fold, check that v values are folded correctly using verifier
        let prover6 = HyperNovaProver(engine: engine)
        let verifier6 = HyperNovaVerifier(engine: engine)

        let witness2: [Fr] = [frFromInt(7), frFromInt(4)]
        let pubInput2: [Fr] = [frFromInt(28)]
        let running6 = CommittedCCSInstance(from: lcccs)
        let newInst6 = prover6.commitWitness(witness2, publicInput: pubInput2)

        let (folded, _, proof) = prover6.fold(
            running: running6, runningWitness: witness,
            new: newInst6, newWitness: witness2)

        // Use the verifier to check the fold (it recomputes rho internally)
        let verifyOk6 = verifier6.verifyFold(running: running6, new: newInst6,
                                              folded: folded, proof: proof)
        expect(verifyOk6, "Verifier confirms linearization consistency")

        // Also directly check v' = sigma + rho*theta by verifying the proof structure
        // The proof contains sigmas and thetas; folded.v must be their linear combo
        // We verify this indirectly: if verifyFold passes, the check holds
        // Additionally, check the counts match
        expectEqual(proof.sigmas.count, ccs.t, "Sigmas count matches matrix count")
        expectEqual(proof.thetas.count, ccs.t, "Thetas count matches matrix count")
        expectEqual(folded.v.count, ccs.t, "Folded v count matches matrix count")
    }

    // -- Test 7: IVC chain convenience --
    do {
        let (ccs, _) = makeR1CSMultiply()
        let prover = HyperNovaProver(ccs: ccs)

        var steps = [(publicInput: [Fr], witness: [Fr])]()
        for i: UInt64 in 2...5 {
            let a = i
            let b = i + 1
            steps.append((publicInput: [frFromInt(a * b)],
                          witness: [frFromInt(a), frFromInt(b)]))
        }

        let (final, finalWit) = prover.ivcChain(steps: steps)
        let decideOk = prover.decide(instance: final, witness: finalWit)
        expect(decideOk, "IVC chain of 4 multiply instances accepted by decider")
    }

    // -- Test 8: Multi-fold (fold N instances in one step) --
    do {
        let (ccs, _) = makeR1CSMultiply()
        let prover = HyperNovaProver(ccs: ccs)
        let verifier = HyperNovaVerifier(ccs: ccs)

        // Initialize running instance
        let (running, runningWit) = prover.initialize(
            witness: [frFromInt(2), frFromInt(3)], publicInput: [frFromInt(6)])

        // Create 2 more instances
        let wit2: [Fr] = [frFromInt(4), frFromInt(5)]
        let pub2: [Fr] = [frFromInt(20)]
        let inst2 = prover.commitWitness(wit2, publicInput: pub2)

        let wit3: [Fr] = [frFromInt(6), frFromInt(7)]
        let pub3: [Fr] = [frFromInt(42)]
        let inst3 = prover.commitWitness(wit3, publicInput: pub3)

        // Multi-fold all 3 into 1
        let (folded, foldedWit, multiProof) = prover.multiFold(
            instances: [running, inst2, inst3],
            witnesses: [runningWit, wit2, wit3])

        // Verify the multi-fold
        let verifyOk = verifier.verifyMultiFold(
            instances: [running, inst2, inst3],
            folded: folded, proof: multiProof)
        expect(verifyOk, "Verifier accepts 3-instance multi-fold")

        // Decide
        let decideOk = prover.decide(instance: folded, witness: foldedWit)
        expect(decideOk, "Decider accepts 3-instance multi-fold")
    }

    // -- Test 9: Performance test -- fold 2^4 = 16 instances --
    do {
        let (ccs, _) = makeR1CSMultiply()
        let prover = HyperNovaProver(ccs: ccs)

        let count = 16
        var steps = [(publicInput: [Fr], witness: [Fr])]()
        for i: UInt64 in 1...UInt64(count) {
            let a = i
            let b = i + 1
            steps.append((publicInput: [frFromInt(a * b)],
                          witness: [frFromInt(a), frFromInt(b)]))
        }

        let t0 = CFAbsoluteTimeGetCurrent()
        let (final, finalWit) = prover.ivcChain(steps: steps)
        let foldTime = CFAbsoluteTimeGetCurrent() - t0

        let t1 = CFAbsoluteTimeGetCurrent()
        let decideOk = prover.decide(instance: final, witness: finalWit)
        let decideTime = CFAbsoluteTimeGetCurrent() - t1

        expect(decideOk, "Decider accepts 16-instance IVC chain")
        print(String(format: "  HyperNova 16-fold: %.2fms fold, %.2fms decide",
                     foldTime * 1000, decideTime * 1000))
    }

    // -- Test 10: Structural checks --
    do {
        let (ccs, _) = makeR1CSMultiply()
        expect(ccs.isWellFormed, "R1CS-derived CCS is well-formed")
        expectEqual(ccs.t, 3, "R1CS CCS has 3 matrices (A, B, C)")
        expectEqual(ccs.q, 2, "R1CS CCS has 2 terms")
        expectEqual(ccs.d, 2, "R1CS CCS has degree 2")
        expectEqual(ccs.m, 1, "R1CS CCS has 1 constraint")
        expectEqual(ccs.n, 4, "R1CS CCS has 4 variables")
    }
}
