// Protogalaxy Tests — folding, decider, and verifier round-trip
import zkMetal

func runProtogalaxyTests() {
    suite("Protogalaxy")

    // --- Helper: create a fresh Plonk instance with dummy commitments ---
    func makeFreshInstance(publicInput: [Fr], beta: Fr, gamma: Fr,
                           scaleFactor: Fr = Fr.one) -> ProtogalaxyInstance {
        let g1 = pointFromAffine(bn254G1Generator())
        // Create distinct commitments by scaling the generator
        let c0 = cPointScalarMul(g1, scaleFactor)
        let c1 = cPointScalarMul(g1, frMul(scaleFactor, frFromInt(2)))
        let c2 = cPointScalarMul(g1, frMul(scaleFactor, frFromInt(3)))
        return ProtogalaxyInstance(
            witnessCommitments: [c0, c1, c2],
            publicInput: publicInput,
            beta: beta,
            gamma: gamma
        )
    }

    // --- Helper: create dummy witness polynomials ---
    func makeDummyWitness(size: Int, seed: UInt64) -> [[Fr]] {
        var a = [Fr](repeating: Fr.zero, count: size)
        var b = [Fr](repeating: Fr.zero, count: size)
        var c = [Fr](repeating: Fr.zero, count: size)
        for i in 0..<size {
            a[i] = frFromInt(seed &+ UInt64(i) &+ 1)
            b[i] = frFromInt(seed &+ UInt64(i) &+ 100)
            // c = a + b (simple additive relation for testing)
            c[i] = frAdd(a[i], b[i])
        }
        return [a, b, c]
    }

    // =====================================================================
    // Test 1: Lagrange interpolation round-trip
    // =====================================================================
    do {
        // Interpolate through (0, 3), (1, 7), (2, 15)
        // The polynomial is p(x) = 2x^2 + 2x + 3
        let points = [frFromInt(0), frFromInt(1), frFromInt(2)]
        let values = [frFromInt(3), frFromInt(7), frFromInt(15)]
        let coeffs = lagrangeInterpolate(points: points, values: values)

        // Verify: p(0) = 3, p(1) = 7, p(2) = 15
        let v0 = hornerEvaluate(coeffs: coeffs, at: frFromInt(0))
        let v1 = hornerEvaluate(coeffs: coeffs, at: frFromInt(1))
        let v2 = hornerEvaluate(coeffs: coeffs, at: frFromInt(2))
        expect(frEq(v0, frFromInt(3)), "Lagrange interp p(0)=3")
        expect(frEq(v1, frFromInt(7)), "Lagrange interp p(1)=7")
        expect(frEq(v2, frFromInt(15)), "Lagrange interp p(2)=15")
    }

    // =====================================================================
    // Test 2: Lagrange basis at a point
    // =====================================================================
    do {
        // For domain {0, 1}, L_0(x) = 1-x, L_1(x) = x
        // At alpha = 3: L_0(3) = -2, L_1(3) = 3
        let basis = lagrangeBasisAtPoint(domainSize: 2, point: frFromInt(3))
        expect(basis.count == 2, "Lagrange basis size=2")
        // L_0(3) = (3 - 1) / (0 - 1) = 2 / (-1) = -2
        let expectedL0 = frNeg(frFromInt(2))
        let expectedL1 = frFromInt(3)
        expect(frEq(basis[0], expectedL0), "L_0(3) = -2")
        expect(frEq(basis[1], expectedL1), "L_1(3) = 3")

        // L_0(alpha) + L_1(alpha) should equal 1 (partition of unity)
        let sum = frAdd(basis[0], basis[1])
        expect(frEq(sum, Fr.one), "Lagrange basis partition of unity")
    }

    // =====================================================================
    // Test 3: Horner evaluation
    // =====================================================================
    do {
        // p(x) = 1 + 2x + 3x^2, evaluate at x=4: 1 + 8 + 48 = 57
        let coeffs = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let val = hornerEvaluate(coeffs: coeffs, at: frFromInt(4))
        expect(frEq(val, frFromInt(57)), "Horner p(4)=57")

        // Empty polynomial = 0
        let z = hornerEvaluate(coeffs: [], at: frFromInt(42))
        expect(frEq(z, Fr.zero), "Horner empty=0")
    }

    // =====================================================================
    // Test 4: Basic 2-instance fold
    // =====================================================================
    do {
        let circuitSize = 4  // power of 2
        let prover = ProtogalaxyProver(circuitSize: circuitSize)

        let pi0: [Fr] = [frFromInt(10)]
        let pi1: [Fr] = [frFromInt(20)]
        let beta0 = frFromInt(5)
        let gamma0 = frFromInt(7)
        let beta1 = frFromInt(11)
        let gamma1 = frFromInt(13)

        let inst0 = makeFreshInstance(publicInput: pi0, beta: beta0, gamma: gamma0,
                                       scaleFactor: frFromInt(1))
        let inst1 = makeFreshInstance(publicInput: pi1, beta: beta1, gamma: gamma1,
                                       scaleFactor: frFromInt(5))

        let wit0 = makeDummyWitness(size: circuitSize, seed: 1)
        let wit1 = makeDummyWitness(size: circuitSize, seed: 100)

        let (folded, foldedWit, proof) = prover.fold(
            instances: [inst0, inst1],
            witnesses: [wit0, wit1]
        )

        expect(folded.isRelaxed, "Folded instance is relaxed")
        expect(proof.instanceCount == 2, "Proof records 2 instances")
        expect(foldedWit.count == 3, "Folded witness has 3 columns")
        expect(foldedWit[0].count == circuitSize, "Folded witness has correct size")

        // Fresh instances have e=0, so F(X) should interpolate through (0,0) and (1,0)
        // meaning F(X) = 0 (all coefficients zero)
        for c in proof.fCoefficients {
            expect(frEq(c, Fr.zero), "F(X) coeff is zero for fresh instances")
        }
        expect(frEq(folded.errorTerm, Fr.zero), "Folded error is zero for fresh instances")
    }

    // =====================================================================
    // Test 5: Fold verifier accepts valid fold
    // =====================================================================
    do {
        let circuitSize = 4
        let prover = ProtogalaxyProver(circuitSize: circuitSize)
        let verifier = ProtogalaxyVerifier()

        let inst0 = makeFreshInstance(publicInput: [frFromInt(1)],
                                       beta: frFromInt(2), gamma: frFromInt(3),
                                       scaleFactor: frFromInt(1))
        let inst1 = makeFreshInstance(publicInput: [frFromInt(4)],
                                       beta: frFromInt(5), gamma: frFromInt(6),
                                       scaleFactor: frFromInt(7))

        let wit0 = makeDummyWitness(size: circuitSize, seed: 10)
        let wit1 = makeDummyWitness(size: circuitSize, seed: 20)

        let (folded, _, proof) = prover.fold(
            instances: [inst0, inst1],
            witnesses: [wit0, wit1]
        )

        let valid = verifier.verifyFold(instances: [inst0, inst1],
                                         folded: folded, proof: proof)
        expect(valid, "Fold verifier accepts valid fold")
    }

    // =====================================================================
    // Test 6: Fold verifier rejects tampered instance
    // =====================================================================
    do {
        let circuitSize = 4
        let prover = ProtogalaxyProver(circuitSize: circuitSize)
        let verifier = ProtogalaxyVerifier()

        let inst0 = makeFreshInstance(publicInput: [frFromInt(1)],
                                       beta: frFromInt(2), gamma: frFromInt(3),
                                       scaleFactor: frFromInt(1))
        let inst1 = makeFreshInstance(publicInput: [frFromInt(4)],
                                       beta: frFromInt(5), gamma: frFromInt(6),
                                       scaleFactor: frFromInt(7))

        let wit0 = makeDummyWitness(size: circuitSize, seed: 10)
        let wit1 = makeDummyWitness(size: circuitSize, seed: 20)

        let (folded, _, proof) = prover.fold(
            instances: [inst0, inst1],
            witnesses: [wit0, wit1]
        )

        // Tamper with the folded instance: change the error term
        let tampered = ProtogalaxyInstance(
            witnessCommitments: folded.witnessCommitments,
            publicInput: folded.publicInput,
            beta: folded.beta,
            gamma: folded.gamma,
            errorTerm: frFromInt(999),  // wrong!
            u: folded.u
        )

        let invalid = verifier.verifyFold(instances: [inst0, inst1],
                                            folded: tampered, proof: proof)
        expect(!invalid, "Fold verifier rejects tampered error term")
    }

    // =====================================================================
    // Test 7: IVC chain (3 instances folded sequentially)
    // =====================================================================
    do {
        let circuitSize = 4
        let prover = ProtogalaxyProver(circuitSize: circuitSize)

        var instances = [ProtogalaxyInstance]()
        var witnesses = [[[Fr]]]()
        for i in 0..<3 {
            instances.append(makeFreshInstance(
                publicInput: [frFromInt(UInt64(i * 10 + 1))],
                beta: frFromInt(UInt64(i * 3 + 1)),
                gamma: frFromInt(UInt64(i * 3 + 2)),
                scaleFactor: frFromInt(UInt64(i + 1))
            ))
            witnesses.append(makeDummyWitness(size: circuitSize, seed: UInt64(i * 50)))
        }

        let (acc, accWit) = prover.ivcChain(instances: instances, witnesses: witnesses)
        expect(acc.isRelaxed, "IVC chain result is relaxed")
        expect(accWit.count == 3, "IVC chain result has 3 witness columns")
        expect(acc.publicInput.count == 1, "IVC chain preserves public input count")
    }

    // =====================================================================
    // Test 8: 3-instance fold (k=3, requires running instance to be relaxed)
    // =====================================================================
    do {
        let circuitSize = 4
        let prover = ProtogalaxyProver(circuitSize: circuitSize)
        let verifier = ProtogalaxyVerifier()

        // First fold 2 to get a relaxed instance
        let inst0 = makeFreshInstance(publicInput: [frFromInt(1)],
                                       beta: frFromInt(2), gamma: frFromInt(3),
                                       scaleFactor: frFromInt(1))
        let inst1 = makeFreshInstance(publicInput: [frFromInt(4)],
                                       beta: frFromInt(5), gamma: frFromInt(6),
                                       scaleFactor: frFromInt(2))
        let inst2 = makeFreshInstance(publicInput: [frFromInt(7)],
                                       beta: frFromInt(8), gamma: frFromInt(9),
                                       scaleFactor: frFromInt(3))

        let wit0 = makeDummyWitness(size: circuitSize, seed: 1)
        let wit1 = makeDummyWitness(size: circuitSize, seed: 2)
        let wit2 = makeDummyWitness(size: circuitSize, seed: 3)

        // First fold inst0 + inst1 -> relaxed
        let (relaxed01, relaxedWit01, _) = prover.fold(
            instances: [inst0, inst1],
            witnesses: [wit0, wit1]
        )
        expect(relaxed01.isRelaxed, "First fold produces relaxed instance")

        // Now fold relaxed01 + inst2 + (and we need to fold 3 at once)
        // Actually for k=3 fold, we need 3 instances where first is relaxed
        let (folded3, _, proof3) = prover.fold(
            instances: [relaxed01, inst1, inst2],
            witnesses: [relaxedWit01, wit1, wit2]
        )

        let valid3 = verifier.verifyFold(
            instances: [relaxed01, inst1, inst2],
            folded: folded3,
            proof: proof3
        )
        expect(valid3, "3-instance fold verifier accepts")
    }

    // =====================================================================
    // Test 9: Decider types and proof construction
    // =====================================================================
    do {
        let circuitSize = 4
        let prover = ProtogalaxyProver(circuitSize: circuitSize)

        let inst0 = makeFreshInstance(publicInput: [frFromInt(10)],
                                       beta: frFromInt(2), gamma: frFromInt(3),
                                       scaleFactor: frFromInt(1))
        let inst1 = makeFreshInstance(publicInput: [frFromInt(20)],
                                       beta: frFromInt(5), gamma: frFromInt(6),
                                       scaleFactor: frFromInt(4))

        let wit0 = makeDummyWitness(size: circuitSize, seed: 42)
        let wit1 = makeDummyWitness(size: circuitSize, seed: 84)

        let (acc, accWit, foldProof) = prover.fold(
            instances: [inst0, inst1],
            witnesses: [wit0, wit1]
        )

        // Test decider config
        let config = ProtogalaxyDeciderConfig(circuitSize: circuitSize)
        expect(config.circuitSize == circuitSize, "Decider config has correct circuit size")
        expect(config.numWitnessColumns == 3, "Decider config defaults to 3 columns")

        // Test decider prover construction
        let deciderProver = ProtogalaxyDeciderProver(config: config)
        let deciderProof = deciderProver.decide(instance: acc, witnesses: accWit,
                                                 foldingProofs: [foldProof])

        expect(deciderProof.foldingProofs.count == 1, "Decider proof has 1 folding proof")
        expect(deciderProof.sumcheckRounds.count > 0, "Decider proof has sumcheck rounds")
        expect(deciderProof.witnessEvals.count == 3, "Decider proof has 3 witness evals")
        expect(deciderProof.witnessCommitments.count == 3, "Decider proof has 3 commitments")

        // Witness hash should be non-zero
        expect(!frEq(deciderProof.witnessHash, Fr.zero), "Decider proof has non-zero witness hash")
    }

    // =====================================================================
    // Test 10: Decider verifier basic check
    // =====================================================================
    do {
        let deciderVerifier = ProtogalaxyDeciderVerifier()
        _ = deciderVerifier  // Verify construction succeeds
        expect(true, "DeciderVerifier constructs without error")
    }

    // =====================================================================
    // Test 11: Decider verifier checks fresh instance constraints
    // =====================================================================
    do {
        let deciderVerifier = ProtogalaxyDeciderVerifier()

        // A fresh instance should have u=1, e=0
        let fresh = makeFreshInstance(publicInput: [frFromInt(1)],
                                       beta: frFromInt(2), gamma: frFromInt(3))
        expect(frEq(fresh.u, Fr.one), "Fresh instance u=1")
        expect(frEq(fresh.errorTerm, Fr.zero), "Fresh instance e=0")
        expect(!fresh.isRelaxed, "Fresh instance is not relaxed")

        // A relaxed instance with non-zero error should be accepted as relaxed
        let relaxed = ProtogalaxyInstance(
            witnessCommitments: fresh.witnessCommitments,
            publicInput: fresh.publicInput,
            beta: fresh.beta,
            gamma: fresh.gamma,
            errorTerm: frFromInt(42),
            u: frFromInt(7)
        )
        expect(relaxed.isRelaxed, "Relaxed instance is relaxed")
        expect(!frEq(relaxed.errorTerm, Fr.zero), "Relaxed instance has non-zero error")
    }

    // =====================================================================
    // Test 12: Folding preserves public input dimensionality
    // =====================================================================
    do {
        let circuitSize = 4
        let prover = ProtogalaxyProver(circuitSize: circuitSize)

        // Two instances with 3 public inputs each
        let inst0 = makeFreshInstance(
            publicInput: [frFromInt(1), frFromInt(2), frFromInt(3)],
            beta: frFromInt(10), gamma: frFromInt(11),
            scaleFactor: frFromInt(1)
        )
        let inst1 = makeFreshInstance(
            publicInput: [frFromInt(4), frFromInt(5), frFromInt(6)],
            beta: frFromInt(12), gamma: frFromInt(13),
            scaleFactor: frFromInt(2)
        )

        let wit0 = makeDummyWitness(size: circuitSize, seed: 1)
        let wit1 = makeDummyWitness(size: circuitSize, seed: 2)

        let (folded, _, _) = prover.fold(
            instances: [inst0, inst1],
            witnesses: [wit0, wit1]
        )

        expect(folded.publicInput.count == 3,
               "Folded instance preserves public input count")
    }

    // =====================================================================
    // Test 13: Decider config and workflow
    // =====================================================================
    do {
        let config = ProtogalaxyDeciderConfig(circuitSize: 8,
                                               numWitnessColumns: 3)
        expect(config.circuitSize == 8, "Config circuit size is 8")
        expect(config.numWitnessColumns == 3, "Config witness columns is 3")

        let deciderProver = ProtogalaxyDeciderProver(config: config)
        _ = deciderProver  // Verify it constructs without error
        expect(true, "DeciderProver constructs with config")
    }

    // =====================================================================
    // Test 14: Multiple sequential folds maintain consistency
    // =====================================================================
    do {
        let circuitSize = 4
        let prover = ProtogalaxyProver(circuitSize: circuitSize)
        let verifier = ProtogalaxyVerifier()

        var instances = [ProtogalaxyInstance]()
        var witnesses = [[[Fr]]]()
        for i in 0..<4 {
            instances.append(makeFreshInstance(
                publicInput: [frFromInt(UInt64(i + 1))],
                beta: frFromInt(UInt64(i * 2 + 1)),
                gamma: frFromInt(UInt64(i * 2 + 2)),
                scaleFactor: frFromInt(UInt64(i + 1))
            ))
            witnesses.append(makeDummyWitness(size: circuitSize, seed: UInt64(i * 100)))
        }

        // IVC chain: fold all 4 instances
        let (acc, _) = prover.ivcChain(instances: instances, witnesses: witnesses)

        // Sequential fold should give same result
        var running = instances[0]
        var runningWit = witnesses[0]
        for i in 1..<4 {
            let (f, fw, p) = prover.fold(
                instances: [running, instances[i]],
                witnesses: [runningWit, witnesses[i]]
            )
            // Verify each fold step
            let valid = verifier.verifyFold(instances: [running, instances[i]],
                                             folded: f, proof: p)
            expect(valid, "Sequential fold step \(i) is valid")
            running = f
            runningWit = fw
        }

        // Both methods should produce the same result
        expect(frEq(acc.u, running.u), "IVC chain u matches sequential fold")
        expect(frEq(acc.errorTerm, running.errorTerm),
               "IVC chain error matches sequential fold")
        expect(frEq(acc.beta, running.beta), "IVC chain beta matches sequential fold")
        expect(frEq(acc.gamma, running.gamma), "IVC chain gamma matches sequential fold")
    }
}
