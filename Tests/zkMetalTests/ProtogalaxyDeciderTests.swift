// ProtogalaxyDeciderTests -- Decider + IVC pipeline tests
import zkMetal

public func runProtogalaxyDeciderTests() {
    suite("Protogalaxy Decider")

    // --- Helper: create a fresh Plonk instance with dummy commitments ---
    func makeFreshInstance(publicInput: [Fr], beta: Fr, gamma: Fr,
                           scaleFactor: Fr = Fr.one) -> ProtogalaxyInstance {
        let g1 = pointFromAffine(bn254G1Generator())
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

    // --- Helper: create a satisfying witness (a + b = c) ---
    func makeSatisfyingWitness(size: Int, seed: UInt64) -> [[Fr]] {
        var a = [Fr](repeating: Fr.zero, count: size)
        var b = [Fr](repeating: Fr.zero, count: size)
        var c = [Fr](repeating: Fr.zero, count: size)
        for i in 0..<size {
            a[i] = frFromInt(seed &+ UInt64(i) &+ 1)
            b[i] = frFromInt(seed &+ UInt64(i) &+ 100)
            c[i] = frAdd(a[i], b[i])  // a + b = c
        }
        return [a, b, c]
    }

    // =====================================================================
    // Test 1: Single-step fold then decide
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

        let wit0 = makeSatisfyingWitness(size: circuitSize, seed: 1)
        let wit1 = makeSatisfyingWitness(size: circuitSize, seed: 50)

        // Fold 2 instances
        let (acc, accWit, _) = prover.fold(
            instances: [inst0, inst1],
            witnesses: [wit0, wit1]
        )
        expect(acc.isRelaxed, "Single-step: folded instance is relaxed")

        // Decide
        let config = ProtogalaxyDeciderConfig(circuitSize: circuitSize)
        let decider = ProtogalaxyDeciderProver(config: config)
        let proof = decider.decide(instance: acc, witnesses: accWit)

        expect(proof.sumcheckRounds.count > 0, "Single-step: proof has sumcheck rounds")
        expect(proof.witnessEvals.count == 3, "Single-step: proof has 3 witness evals")
        expect(proof.witnessCommitments.count == 3, "Single-step: proof has 3 commitments")
    }

    // =====================================================================
    // Test 2: Multi-step fold (5 instances) then decide
    // =====================================================================
    do {
        let circuitSize = 4
        let prover = ProtogalaxyProver(circuitSize: circuitSize)

        var instances = [ProtogalaxyInstance]()
        var witnesses = [[[Fr]]]()
        for i in 0..<5 {
            instances.append(makeFreshInstance(
                publicInput: [frFromInt(UInt64(i * 10 + 1))],
                beta: frFromInt(UInt64(i * 3 + 1)),
                gamma: frFromInt(UInt64(i * 3 + 2)),
                scaleFactor: frFromInt(UInt64(i + 1))
            ))
            witnesses.append(makeSatisfyingWitness(size: circuitSize, seed: UInt64(i * 50)))
        }

        // IVC chain: fold all 5 instances
        let (acc, accWit) = prover.ivcChain(instances: instances, witnesses: witnesses)
        expect(acc.isRelaxed, "Multi-step: accumulated instance is relaxed")

        // Decide
        let config = ProtogalaxyDeciderConfig(circuitSize: circuitSize)
        let decider = ProtogalaxyDeciderProver(config: config)
        let proof = decider.decide(instance: acc, witnesses: accWit)

        expect(proof.sumcheckRounds.count > 0, "Multi-step: proof has sumcheck rounds")
        expect(proof.witnessEvals.count == 3, "Multi-step: proof has 3 witness evals")

        // The number of sumcheck rounds should be log2(circuitSize)
        let expectedRounds = 2  // log2(4) = 2
        expect(proof.sumcheckRounds.count == expectedRounds,
               "Multi-step: sumcheck has \(expectedRounds) rounds")
    }

    // =====================================================================
    // Test 3: Verifier accepts valid proof
    // =====================================================================
    do {
        let circuitSize = 4
        let prover = ProtogalaxyProver(circuitSize: circuitSize)

        let inst0 = makeFreshInstance(publicInput: [frFromInt(1)],
                                       beta: frFromInt(2), gamma: frFromInt(3),
                                       scaleFactor: frFromInt(1))
        let inst1 = makeFreshInstance(publicInput: [frFromInt(4)],
                                       beta: frFromInt(5), gamma: frFromInt(6),
                                       scaleFactor: frFromInt(7))

        let wit0 = makeSatisfyingWitness(size: circuitSize, seed: 10)
        let wit1 = makeSatisfyingWitness(size: circuitSize, seed: 20)

        let (acc, accWit, _) = prover.fold(
            instances: [inst0, inst1],
            witnesses: [wit0, wit1]
        )

        let config = ProtogalaxyDeciderConfig(circuitSize: circuitSize)
        let decider = ProtogalaxyDeciderProver(config: config)
        let proof = decider.decide(instance: acc, witnesses: accWit)

        let verifier = ProtogalaxyDeciderVerifier()
        let valid = verifier.verify(proof: proof)
        expect(valid, "Verifier accepts valid decider proof")
    }

    // =====================================================================
    // Test 4: Verifier rejects tampered proof
    // =====================================================================
    do {
        let circuitSize = 4
        let prover = ProtogalaxyProver(circuitSize: circuitSize)

        let inst0 = makeFreshInstance(publicInput: [frFromInt(1)],
                                       beta: frFromInt(2), gamma: frFromInt(3),
                                       scaleFactor: frFromInt(1))
        let inst1 = makeFreshInstance(publicInput: [frFromInt(4)],
                                       beta: frFromInt(5), gamma: frFromInt(6),
                                       scaleFactor: frFromInt(7))

        let wit0 = makeSatisfyingWitness(size: circuitSize, seed: 10)
        let wit1 = makeSatisfyingWitness(size: circuitSize, seed: 20)

        let (acc, accWit, _) = prover.fold(
            instances: [inst0, inst1],
            witnesses: [wit0, wit1]
        )

        let config = ProtogalaxyDeciderConfig(circuitSize: circuitSize)
        let decider = ProtogalaxyDeciderProver(config: config)
        let proof = decider.decide(instance: acc, witnesses: accWit)

        // Tamper with a sumcheck round polynomial
        var tamperedRounds = proof.sumcheckRounds
        if !tamperedRounds.isEmpty {
            let (s0, s1, s2) = tamperedRounds[0]
            tamperedRounds[0] = (frAdd(s0, Fr.one), s1, s2)  // corrupt s(0)
        }
        let tamperedProof = ProtogalaxyDeciderProof(
            witnessCommitments: proof.witnessCommitments,
            sumcheckRounds: tamperedRounds,
            witnessEvals: proof.witnessEvals,
            accumulatedInstance: proof.accumulatedInstance,
            foldingProofs: proof.foldingProofs,
            witnessHash: proof.witnessHash
        )

        let verifier = ProtogalaxyDeciderVerifier()
        let invalid = verifier.verify(proof: tamperedProof)
        expect(!invalid, "Verifier rejects tampered sumcheck round")

        // Also tamper with witness evaluations
        var tamperedEvals = proof.witnessEvals
        if !tamperedEvals.isEmpty {
            tamperedEvals[0] = frAdd(tamperedEvals[0], Fr.one)  // corrupt a_eval
        }
        let tamperedProof2 = ProtogalaxyDeciderProof(
            witnessCommitments: proof.witnessCommitments,
            sumcheckRounds: proof.sumcheckRounds,
            witnessEvals: tamperedEvals,
            accumulatedInstance: proof.accumulatedInstance,
            foldingProofs: proof.foldingProofs,
            witnessHash: proof.witnessHash
        )
        let invalid2 = verifier.verify(proof: tamperedProof2)
        expect(!invalid2, "Verifier rejects tampered witness evaluations")
    }

    // =====================================================================
    // Test 5: IVC pipeline: 3 steps + prove + verify
    // =====================================================================
    do {
        let circuitSize = 4
        let ivc = ProtogalaxyIVC(circuitSize: circuitSize)

        // Step 1
        let inst0 = makeFreshInstance(publicInput: [frFromInt(1)],
                                       beta: frFromInt(2), gamma: frFromInt(3),
                                       scaleFactor: frFromInt(1))
        let wit0 = makeSatisfyingWitness(size: circuitSize, seed: 10)
        let state0 = ivc.step(instance: inst0, witness: wit0)
        expect(state0.stepCount == 1, "IVC step 1: count = 1")

        // Step 2
        let inst1 = makeFreshInstance(publicInput: [frFromInt(4)],
                                       beta: frFromInt(5), gamma: frFromInt(6),
                                       scaleFactor: frFromInt(2))
        let wit1 = makeSatisfyingWitness(size: circuitSize, seed: 20)
        let state1 = ivc.step(instance: inst1, witness: wit1)
        expect(state1.stepCount == 2, "IVC step 2: count = 2")
        expect(state1.instance.isRelaxed, "IVC step 2: instance is relaxed after fold")

        // Step 3
        let inst2 = makeFreshInstance(publicInput: [frFromInt(7)],
                                       beta: frFromInt(8), gamma: frFromInt(9),
                                       scaleFactor: frFromInt(3))
        let wit2 = makeSatisfyingWitness(size: circuitSize, seed: 30)
        let state2 = ivc.step(instance: inst2, witness: wit2)
        expect(state2.stepCount == 3, "IVC step 3: count = 3")

        // Prove
        let proof = ivc.prove()
        expect(proof.sumcheckRounds.count > 0, "IVC prove: proof has sumcheck rounds")
        expect(proof.foldingProofs.count == 2, "IVC prove: 2 folding proofs for 3 steps")

        // Verify (sumcheck only)
        let valid = ProtogalaxyIVC.verify(proof: proof)
        expect(valid, "IVC verify: accepts valid proof")

        // Verify with full chain
        let validChain = ProtogalaxyIVC.verifyIVC(
            proof: proof,
            originalInstances: ivc.instanceHistory
        )
        expect(validChain, "IVC verify chain: accepts valid chain")
    }

    // =====================================================================
    // Test 6: IVC state tracking and reset
    // =====================================================================
    do {
        let circuitSize = 4
        let ivc = ProtogalaxyIVC(circuitSize: circuitSize)

        expect(ivc.currentState == nil, "IVC: no state before first step")
        expect(ivc.currentStepCount == 0, "IVC: step count = 0 initially")

        let inst = makeFreshInstance(publicInput: [frFromInt(1)],
                                      beta: frFromInt(2), gamma: frFromInt(3))
        let wit = makeSatisfyingWitness(size: circuitSize, seed: 1)
        ivc.step(instance: inst, witness: wit)

        expect(ivc.currentState != nil, "IVC: has state after first step")
        expect(ivc.currentStepCount == 1, "IVC: step count = 1")
        expect(ivc.instanceHistory.count == 1, "IVC: 1 instance in history")

        // Reset
        ivc.reset()
        expect(ivc.currentState == nil, "IVC: no state after reset")
        expect(ivc.currentStepCount == 0, "IVC: step count = 0 after reset")
        expect(ivc.instanceHistory.count == 0, "IVC: empty history after reset")
    }

    // =====================================================================
    // Test 7: Decider config validation
    // =====================================================================
    do {
        let config = ProtogalaxyDeciderConfig(circuitSize: 8, numWitnessColumns: 3)
        expect(config.circuitSize == 8, "DeciderConfig: circuitSize = 8")
        expect(config.numWitnessColumns == 3, "DeciderConfig: numWitnessColumns = 3")

        let decider = ProtogalaxyDeciderProver(config: config)
        _ = decider  // verify construction succeeds
        expect(true, "DeciderProver: constructs without error")
    }

    // =====================================================================
    // Test 8: Larger circuit size (8 gates)
    // =====================================================================
    do {
        let circuitSize = 8
        let prover = ProtogalaxyProver(circuitSize: circuitSize)

        let inst0 = makeFreshInstance(publicInput: [frFromInt(1)],
                                       beta: frFromInt(2), gamma: frFromInt(3),
                                       scaleFactor: frFromInt(1))
        let inst1 = makeFreshInstance(publicInput: [frFromInt(4)],
                                       beta: frFromInt(5), gamma: frFromInt(6),
                                       scaleFactor: frFromInt(7))

        let wit0 = makeSatisfyingWitness(size: circuitSize, seed: 10)
        let wit1 = makeSatisfyingWitness(size: circuitSize, seed: 20)

        let (acc, accWit, _) = prover.fold(
            instances: [inst0, inst1],
            witnesses: [wit0, wit1]
        )

        let config = ProtogalaxyDeciderConfig(circuitSize: circuitSize)
        let decider = ProtogalaxyDeciderProver(config: config)
        let proof = decider.decide(instance: acc, witnesses: accWit)

        // log2(8) = 3 rounds
        expect(proof.sumcheckRounds.count == 3, "Large circuit: 3 sumcheck rounds")

        let verifier = ProtogalaxyDeciderVerifier()
        let valid = verifier.verify(proof: proof)
        expect(valid, "Large circuit: verifier accepts valid proof")
    }

    // =====================================================================
    // Test 9: Multiple public inputs
    // =====================================================================
    do {
        let circuitSize = 4
        let prover = ProtogalaxyProver(circuitSize: circuitSize)

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

        let wit0 = makeSatisfyingWitness(size: circuitSize, seed: 1)
        let wit1 = makeSatisfyingWitness(size: circuitSize, seed: 2)

        let (acc, accWit, _) = prover.fold(
            instances: [inst0, inst1],
            witnesses: [wit0, wit1]
        )

        let config = ProtogalaxyDeciderConfig(circuitSize: circuitSize)
        let decider = ProtogalaxyDeciderProver(config: config)
        let proof = decider.decide(instance: acc, witnesses: accWit)

        let verifier = ProtogalaxyDeciderVerifier()
        let valid = verifier.verify(proof: proof)
        expect(valid, "Multiple PI: verifier accepts")
        expect(proof.accumulatedInstance.publicInput.count == 3,
               "Multiple PI: preserved in proof")
    }

    // =====================================================================
    // Test 10: Witness hash binding
    // =====================================================================
    do {
        let circuitSize = 4
        let prover = ProtogalaxyProver(circuitSize: circuitSize)

        let inst0 = makeFreshInstance(publicInput: [frFromInt(1)],
                                       beta: frFromInt(2), gamma: frFromInt(3),
                                       scaleFactor: frFromInt(1))
        let inst1 = makeFreshInstance(publicInput: [frFromInt(4)],
                                       beta: frFromInt(5), gamma: frFromInt(6),
                                       scaleFactor: frFromInt(7))

        let wit0 = makeSatisfyingWitness(size: circuitSize, seed: 10)
        let wit1 = makeSatisfyingWitness(size: circuitSize, seed: 20)

        let (acc, accWit, _) = prover.fold(
            instances: [inst0, inst1],
            witnesses: [wit0, wit1]
        )

        let config = ProtogalaxyDeciderConfig(circuitSize: circuitSize)
        let decider = ProtogalaxyDeciderProver(config: config)
        let proof = decider.decide(instance: acc, witnesses: accWit)

        // Witness hash should be non-zero
        expect(!frEq(proof.witnessHash, Fr.zero), "Witness hash is non-zero")

        // Different witness should produce different hash
        let wit2 = makeSatisfyingWitness(size: circuitSize, seed: 99)
        let proof2 = decider.decide(instance: acc, witnesses: [wit2[0], wit2[1], wit2[2]])
        expect(!frEq(proof.witnessHash, proof2.witnessHash),
               "Different witness produces different hash")
    }
}
