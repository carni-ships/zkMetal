import zkMetal

func runDataParallelTests() {
    suite("Data-Parallel GKR")

    // Helper: deterministic pseudo-random inputs
    func pseudoRandomInputs(_ n: Int, seed: UInt64 = 0xDEAD_BEEF_CAFE_1234) -> [Fr] {
        var rng = seed
        return (0..<n).map { _ in
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return frFromInt(rng >> 32)
        }
    }

    // --- DataParallelCircuit basic tests ---

    // Test: circuit construction and evaluation
    do {
        let template = LayeredCircuit.repeatedHashCircuit(logWidth: 2, depth: 2)
        let inputs1 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let inputs2 = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]

        var dpCircuit = DataParallelCircuit(
            template: template, instances: 2, inputs: [inputs1, inputs2])

        let outputs = dpCircuit.evaluateAll()
        expectEqual(outputs.count, 2, "DP circuit: 2 instance outputs")

        // Verify against individual evaluations
        let out1 = template.evaluateOutput(inputs: inputs1)
        let out2 = template.evaluateOutput(inputs: inputs2)
        for i in 0..<out1.count {
            let match = frToInt(frSub(outputs[0][i], out1[i]))
            expectEqual(match[0], 0, "DP circuit: instance 0 output[\(i)] matches")
        }
        for i in 0..<out2.count {
            let match = frToInt(frSub(outputs[1][i], out2[i]))
            expectEqual(match[0], 0, "DP circuit: instance 1 output[\(i)] matches")
        }
    }

    // Test: wiring MLEs are shared (same object for same layer)
    do {
        let template = LayeredCircuit.repeatedHashCircuit(logWidth: 2, depth: 2)
        let inputs1 = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let inputs2 = [frFromInt(50), frFromInt(60), frFromInt(70), frFromInt(80)]

        var dpCircuit = DataParallelCircuit(
            template: template, instances: 2, inputs: [inputs1, inputs2])
        dpCircuit.evaluateAll()

        let addMLE0a = dpCircuit.addWiringMLE(layer: 0)
        let addMLE0b = dpCircuit.addWiringMLE(layer: 0)
        // Same evaluations (cached)
        expectEqual(addMLE0a.numVars, addMLE0b.numVars, "DP wiring: cached MLE same numVars")
        let match = frToInt(frSub(addMLE0a.evals[0], addMLE0b.evals[0]))
        expectEqual(match[0], 0, "DP wiring: cached MLE same evals")
    }

    // --- Test the WIP engine (known working) for comparison ---
    do {
        let sub = SubCircuit.repeatedSquaring(rounds: 3)
        let inp1 = [frFromInt(3)]
        let inp2 = [frFromInt(5)]
        let uniformCircuit = UniformCircuit(subCircuit: sub, inputs: [inp1, inp2])

        let proverT = Transcript(label: "dp-engine-test")
        let engine = DataParallelEngine()
        let proof = engine.prove(circuit: uniformCircuit, transcript: proverT)

        let verifierT = Transcript(label: "dp-engine-test")
        let valid = engine.verify(
            subCircuit: sub, numInstances: 2,
            inputs: [inp1, inp2], proof: proof,
            transcript: verifierT)
        expectEqual(valid, true, "DP engine baseline: squaring proof verifies")
    }

    // --- Minimal debug test ---
    do {
        // Simplest possible: 1 instance, 1 layer, 2 gates
        let layer = CircuitLayer(gates: [
            Gate(type: .add, leftInput: 0, rightInput: 1),
            Gate(type: .mul, leftInput: 0, rightInput: 1),
        ])
        let template = LayeredCircuit(layers: [layer])
        let inputs = [frFromInt(3), frFromInt(5)]

        var dpCircuit = DataParallelCircuit(
            template: template, instances: 1, inputs: [inputs])
        dpCircuit.evaluateAll()

        let out = dpCircuit.instanceOutputs![0]
        expectEqual(frToInt(out[0])[0], 8, "DP minimal: 3+5=8")
        expectEqual(frToInt(out[1])[0], 15, "DP minimal: 3*5=15")

        let proverT = Transcript(label: "dp-minimal")
        let prover = DataParallelProver()
        let proof = prover.prove(circuit: &dpCircuit, transcript: proverT)

        let verifierT = Transcript(label: "dp-minimal")
        let verifier = DataParallelVerifier()
        let valid = verifier.verify(
            template: template, numInstances: 1,
            inputs: [inputs], proof: proof, transcript: verifierT)
        expectEqual(valid, true, "DP minimal 1-inst 1-layer: proof verifies")
    }

    // --- Prover + Verifier round-trip tests ---

    // Test: 2 instances of repeated squaring (depth 3)
    do {
        let sub = SubCircuit.repeatedSquaring(rounds: 3)
        let template = sub.toLayeredCircuit()

        let inp1 = [frFromInt(3)]  // 3 -> 9 -> 81 -> 6561
        let inp2 = [frFromInt(5)]  // 5 -> 25 -> 625 -> 390625

        var dpCircuit = DataParallelCircuit(
            template: template, instances: 2, inputs: [inp1, inp2])
        dpCircuit.evaluateAll()

        // Check outputs
        let out1 = dpCircuit.instanceOutputs![0]
        let out2 = dpCircuit.instanceOutputs![1]
        expectEqual(frToInt(out1[0])[0], 6561, "DP squaring: 3^8 = 6561")
        expectEqual(frToInt(out2[0])[0], 390625, "DP squaring: 5^8 = 390625")

        // Prove
        let proverTranscript = Transcript(label: "dp-test-squaring")
        let prover = DataParallelProver()
        let proof = prover.prove(circuit: &dpCircuit, transcript: proverTranscript)

        expectEqual(proof.layerProofs.count, 3, "DP proof: 3 layer proofs")
        expectEqual(proof.allOutputs.count, 2, "DP proof: 2 output sets")

        // Verify
        let verifierTranscript = Transcript(label: "dp-test-squaring")
        let verifier = DataParallelVerifier()
        let valid = verifier.verify(
            template: template,
            numInstances: 2,
            inputs: [inp1, inp2],
            proof: proof,
            transcript: verifierTranscript)
        expectEqual(valid, true, "DP squaring: proof verifies")
    }

    // Test: 4 instances of hash-like circuit
    do {
        let sub = SubCircuit.hashLike(logWidth: 2, numRounds: 2)
        let template = sub.toLayeredCircuit()
        let width = 4

        var allInputs = [[Fr]]()
        for i in 0..<4 {
            allInputs.append(pseudoRandomInputs(width, seed: UInt64(i + 1) &* 0xABCD))
        }

        var dpCircuit = DataParallelCircuit(
            template: template, instances: 4, inputs: allInputs)
        dpCircuit.evaluateAll()

        // Verify each instance output matches individual evaluation
        for i in 0..<4 {
            let expected = template.evaluateOutput(inputs: allInputs[i])
            let actual = dpCircuit.instanceOutputs![i]
            for j in 0..<min(expected.count, actual.count) {
                let diff = frToInt(frSub(expected[j], actual[j]))
                expectEqual(diff[0], 0, "DP hash 4-inst: instance \(i) output[\(j)]")
            }
        }

        // Prove and verify
        let proverT = Transcript(label: "dp-test-hash4")
        let prover = DataParallelProver()
        let proof = prover.prove(circuit: &dpCircuit, transcript: proverT)

        let verifierT = Transcript(label: "dp-test-hash4")
        let verifier = DataParallelVerifier()
        let valid = verifier.verify(
            template: template,
            numInstances: 4,
            inputs: allInputs,
            proof: proof,
            transcript: verifierT)
        expectEqual(valid, true, "DP hash 4 instances: proof verifies")
    }

    // Test: soundness — tampered output should fail verification
    do {
        let sub = SubCircuit.repeatedSquaring(rounds: 2)
        let template = sub.toLayeredCircuit()
        let inp1 = [frFromInt(2)]
        let inp2 = [frFromInt(3)]

        var dpCircuit = DataParallelCircuit(
            template: template, instances: 2, inputs: [inp1, inp2])
        dpCircuit.evaluateAll()

        let proverT = Transcript(label: "dp-test-soundness")
        let prover = DataParallelProver()
        let proof = prover.prove(circuit: &dpCircuit, transcript: proverT)

        // Tamper with outputs
        var tamperedOutputs = proof.allOutputs
        tamperedOutputs[0] = [frFromInt(999)]
        let tamperedProof = DataParallelGKRProof(
            layerProofs: proof.layerProofs, allOutputs: tamperedOutputs)

        let verifierT = Transcript(label: "dp-test-soundness")
        let verifier = DataParallelVerifier()
        let valid = verifier.verify(
            template: template,
            numInstances: 2,
            inputs: [inp1, inp2],
            proof: tamperedProof,
            transcript: verifierT)
        expectEqual(valid, false, "DP soundness: tampered proof rejected")
    }

    // Test: single instance (degenerate case, N=1)
    do {
        let template = LayeredCircuit.innerProductCircuit(size: 4)
        // Inner product of [1,2,3,4] . [5,6,7,8] = 1*5 + 2*6 + 3*7 + 4*8 = 5+12+21+32 = 70
        let inputs = [frFromInt(1), frFromInt(5), frFromInt(2), frFromInt(6),
                      frFromInt(3), frFromInt(7), frFromInt(4), frFromInt(8)]

        var dpCircuit = DataParallelCircuit(
            template: template, instances: 1, inputs: [inputs])
        let outputs = dpCircuit.evaluateAll()
        expectEqual(frToInt(outputs[0][0])[0], 70, "DP inner product: 1*5+2*6+3*7+4*8 = 70")

        let proverT = Transcript(label: "dp-test-single")
        let prover = DataParallelProver()
        let proof = prover.prove(circuit: &dpCircuit, transcript: proverT)

        let verifierT = Transcript(label: "dp-test-single")
        let verifier = DataParallelVerifier()
        let valid = verifier.verify(
            template: template,
            numInstances: 1,
            inputs: [inputs],
            proof: proof,
            transcript: verifierT)
        expectEqual(valid, true, "DP single instance: proof verifies")
    }

    // Test: existing DataParallelEngine still works (backward compat)
    do {
        let sub = SubCircuit.repeatedSquaring(rounds: 2)
        let inputs: [[Fr]] = [[frFromInt(2)], [frFromInt(3)]]
        let uniformCircuit = UniformCircuit(subCircuit: sub, inputs: inputs)

        let proverT = Transcript(label: "dp-engine-compat")
        let engine = DataParallelEngine()
        let proof = engine.prove(circuit: uniformCircuit, transcript: proverT)

        let verifierT = Transcript(label: "dp-engine-compat")
        let valid = engine.verify(
            subCircuit: sub, numInstances: 2,
            inputs: inputs, proof: proof,
            transcript: verifierT)
        expectEqual(valid, true, "DP engine compat: original API still works")
    }
}
