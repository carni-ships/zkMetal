import zkMetal

func runGKRTests() {
    suite("GKR Protocol")

    // Helper: deterministic pseudo-random inputs
    func pseudoRandomInputs(_ n: Int, seed: UInt64 = 0xCAFE_BABE_DEAD_BEEF) -> [Fr] {
        var rng = seed
        return (0..<n).map { _ in
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return frFromInt(rng >> 32)
        }
    }

    // --- Circuit evaluation tests ---

    // Test: single add gate
    do {
        let layer = CircuitLayer(gates: [Gate(type: .add, leftInput: 0, rightInput: 1)])
        let circuit = LayeredCircuit(layers: [layer])
        let inputs = [frFromInt(3), frFromInt(5)]
        let output = circuit.evaluateOutput(inputs: inputs)
        expectEqual(frToInt(output[0])[0], UInt64(8), "Add gate: 3 + 5 = 8")
    }

    // Test: single mul gate
    do {
        let layer = CircuitLayer(gates: [Gate(type: .mul, leftInput: 0, rightInput: 1)])
        let circuit = LayeredCircuit(layers: [layer])
        let inputs = [frFromInt(3), frFromInt(5)]
        let output = circuit.evaluateOutput(inputs: inputs)
        expectEqual(frToInt(output[0])[0], UInt64(15), "Mul gate: 3 * 5 = 15")
    }

    // Test: two gates (add + mul)
    do {
        let layer = CircuitLayer(gates: [
            Gate(type: .add, leftInput: 0, rightInput: 1),
            Gate(type: .mul, leftInput: 0, rightInput: 1),
        ])
        let circuit = LayeredCircuit(layers: [layer])
        let inputs = [frFromInt(3), frFromInt(5)]
        let output = circuit.evaluateOutput(inputs: inputs)
        expectEqual(frToInt(output[0])[0], UInt64(8), "Two gates: add output = 8")
        expectEqual(frToInt(output[1])[0], UInt64(15), "Two gates: mul output = 15")
    }

    // Test: two-layer circuit
    do {
        let l0 = CircuitLayer(gates: [
            Gate(type: .add, leftInput: 0, rightInput: 1),
            Gate(type: .mul, leftInput: 0, rightInput: 1),
        ])
        let l1 = CircuitLayer(gates: [Gate(type: .mul, leftInput: 0, rightInput: 1)])
        let circuit = LayeredCircuit(layers: [l0, l1])
        let inputs = [frFromInt(3), frFromInt(5)]
        let output = circuit.evaluateOutput(inputs: inputs)
        // (3+5) * (3*5) = 8 * 15 = 120
        expectEqual(frToInt(output[0])[0], UInt64(120), "Two-layer: (3+5)*(3*5) = 120")
    }

    // --- MultilinearPoly tests ---

    // Test: evaluate MLE at boolean point matches stored value
    do {
        let evals = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let mle = MultilinearPoly(numVars: 2, evals: evals)
        let v00 = mle.evaluate(at: [Fr.zero, Fr.zero])
        let v01 = mle.evaluate(at: [Fr.zero, Fr.one])
        let v10 = mle.evaluate(at: [Fr.one, Fr.zero])
        let v11 = mle.evaluate(at: [Fr.one, Fr.one])
        expectEqual(frToInt(v00)[0], UInt64(10), "MLE(0,0) = 10")
        expectEqual(frToInt(v01)[0], UInt64(20), "MLE(0,1) = 20")
        expectEqual(frToInt(v10)[0], UInt64(30), "MLE(1,0) = 30")
        expectEqual(frToInt(v11)[0], UInt64(40), "MLE(1,1) = 40")
    }

    // Test: fixVariable reduces dimension
    do {
        let evals = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let mle = MultilinearPoly(numVars: 2, evals: evals)
        // Fix variable 0 to 0 -> should give [10, 20]
        let fixed0 = mle.fixVariable(Fr.zero)
        expectEqual(fixed0.numVars, 1, "fixVariable reduces numVars")
        expectEqual(frToInt(fixed0.evaluate(at: [Fr.zero]))[0], UInt64(10), "fix(0) eval(0) = 10")
        expectEqual(frToInt(fixed0.evaluate(at: [Fr.one]))[0], UInt64(20), "fix(0) eval(1) = 20")
    }

    // Test: eqPoly at boolean points
    do {
        let eq = MultilinearPoly.eqPoly(point: [Fr.one, Fr.zero])
        // eq(r, x) at x=(1,0) should be 1 when r=(1,0)
        expectEqual(frToInt(eq[2])[0], UInt64(1), "eq((1,0), (1,0)) = 1")
        expect(eq[0].isZero, "eq((1,0), (0,0)) = 0")
        expect(eq[1].isZero, "eq((1,0), (0,1)) = 0")
        expect(eq[3].isZero, "eq((1,0), (1,1)) = 0")
    }

    // --- GKR prove/verify tests ---

    // Test: 1-layer (add + mul)
    do {
        let layer = CircuitLayer(gates: [
            Gate(type: .add, leftInput: 0, rightInput: 1),
            Gate(type: .mul, leftInput: 0, rightInput: 1),
        ])
        let circuit = LayeredCircuit(layers: [layer])
        let inputs = [frFromInt(3), frFromInt(5)]
        let output = circuit.evaluateOutput(inputs: inputs)
        let engine = GKREngine(circuit: circuit)
        let pt = Transcript(label: "test-1l", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)
        let vt = Transcript(label: "test-1l", backend: .keccak256)
        let ok = engine.verify(inputs: inputs, output: output, proof: proof, transcript: vt)
        expect(ok, "GKR 1-layer prove/verify")
    }

    // Test: 2-layer
    do {
        let l0 = CircuitLayer(gates: [
            Gate(type: .add, leftInput: 0, rightInput: 1),
            Gate(type: .mul, leftInput: 0, rightInput: 1),
        ])
        let l1 = CircuitLayer(gates: [Gate(type: .mul, leftInput: 0, rightInput: 1)])
        let circuit = LayeredCircuit(layers: [l0, l1])
        let inputs = [frFromInt(3), frFromInt(5)]
        let output = circuit.evaluateOutput(inputs: inputs)
        let engine = GKREngine(circuit: circuit)
        let pt = Transcript(label: "test-2l", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)
        let vt = Transcript(label: "test-2l", backend: .keccak256)
        let ok = engine.verify(inputs: inputs, output: output, proof: proof, transcript: vt)
        expect(ok, "GKR 2-layer prove/verify")
    }

    // Test: hash circuit (width=8, depth=2)
    do {
        let circuit = LayeredCircuit.repeatedHashCircuit(logWidth: 3, depth: 2)
        let inputs = pseudoRandomInputs(8)
        let output = circuit.evaluateOutput(inputs: inputs)
        let engine = GKREngine(circuit: circuit)
        let pt = Transcript(label: "hash", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)
        let vt = Transcript(label: "hash", backend: .keccak256)
        let ok = engine.verify(inputs: inputs, output: output, proof: proof, transcript: vt)
        expect(ok, "GKR hash circuit 2^3 d=2")
    }

    // Test: hash circuit (width=16, depth=4)
    do {
        let circuit = LayeredCircuit.repeatedHashCircuit(logWidth: 4, depth: 4)
        let inputs = pseudoRandomInputs(16)
        let output = circuit.evaluateOutput(inputs: inputs)
        let engine = GKREngine(circuit: circuit)
        let pt = Transcript(label: "hash4", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)
        let vt = Transcript(label: "hash4", backend: .keccak256)
        let ok = engine.verify(inputs: inputs, output: output, proof: proof, transcript: vt)
        expect(ok, "GKR hash circuit 2^4 d=4")
    }

    // Test: inner product circuit
    do {
        let circuit = LayeredCircuit.innerProductCircuit(size: 4)
        let inputs = pseudoRandomInputs(8, seed: 42)
        let output = circuit.evaluateOutput(inputs: inputs)
        let engine = GKREngine(circuit: circuit)
        let pt = Transcript(label: "ip", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)
        let vt = Transcript(label: "ip", backend: .keccak256)
        let ok = engine.verify(inputs: inputs, output: output, proof: proof, transcript: vt)
        expect(ok, "GKR inner product n=4")
    }

    // Test: larger inner product
    do {
        let circuit = LayeredCircuit.innerProductCircuit(size: 16)
        let inputs = pseudoRandomInputs(32, seed: 99)
        let output = circuit.evaluateOutput(inputs: inputs)
        let engine = GKREngine(circuit: circuit)
        let pt = Transcript(label: "ip16", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)
        let vt = Transcript(label: "ip16", backend: .keccak256)
        let ok = engine.verify(inputs: inputs, output: output, proof: proof, transcript: vt)
        expect(ok, "GKR inner product n=16")
    }

    // Test: soundness - wrong output should fail
    do {
        let layer = CircuitLayer(gates: [
            Gate(type: .add, leftInput: 0, rightInput: 1),
            Gate(type: .mul, leftInput: 0, rightInput: 1),
        ])
        let circuit = LayeredCircuit(layers: [layer])
        let inputs = [frFromInt(3), frFromInt(5)]
        let output = circuit.evaluateOutput(inputs: inputs)
        let engine = GKREngine(circuit: circuit)
        let pt = Transcript(label: "sound", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)

        // Tamper with output
        var badOutput = output
        badOutput[0] = frFromInt(999)
        let vt = Transcript(label: "sound", backend: .keccak256)
        let ok = engine.verify(inputs: inputs, output: badOutput, proof: proof, transcript: vt)
        expect(!ok, "GKR rejects wrong output")
    }

    // Test: soundness - wrong inputs should fail
    do {
        let layer = CircuitLayer(gates: [
            Gate(type: .add, leftInput: 0, rightInput: 1),
            Gate(type: .mul, leftInput: 0, rightInput: 1),
        ])
        let circuit = LayeredCircuit(layers: [layer])
        let inputs = [frFromInt(3), frFromInt(5)]
        let output = circuit.evaluateOutput(inputs: inputs)
        let engine = GKREngine(circuit: circuit)
        let pt = Transcript(label: "sound2", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)

        // Verify with wrong inputs
        let badInputs = [frFromInt(3), frFromInt(7)]
        let vt = Transcript(label: "sound2", backend: .keccak256)
        let ok = engine.verify(inputs: badInputs, output: output, proof: proof, transcript: vt)
        expect(!ok, "GKR rejects wrong inputs")
    }

    // Test: proof structure
    do {
        let circuit = LayeredCircuit.repeatedHashCircuit(logWidth: 3, depth: 3)
        let inputs = pseudoRandomInputs(8, seed: 77)
        let engine = GKREngine(circuit: circuit)
        let pt = Transcript(label: "struct", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)
        expectEqual(proof.layerProofs.count, 3, "GKR proof has one layer proof per layer")
        // Each layer: nIn = 3 (logWidth), totalVars = 2*3 = 6
        for (i, lp) in proof.layerProofs.enumerated() {
            expectEqual(lp.sumcheckMsgs.count, 6, "Layer \(i) sumcheck has 2*nIn rounds")
        }
    }

    // Test: transcript determinism
    do {
        let circuit = LayeredCircuit.repeatedHashCircuit(logWidth: 3, depth: 2)
        let inputs = pseudoRandomInputs(8, seed: 123)
        let engine = GKREngine(circuit: circuit)
        let pt1 = Transcript(label: "det", backend: .keccak256)
        let proof1 = engine.prove(inputs: inputs, transcript: pt1)
        let pt2 = Transcript(label: "det", backend: .keccak256)
        let proof2 = engine.prove(inputs: inputs, transcript: pt2)

        // Same inputs + same transcript label = same proof
        var match = true
        for i in 0..<proof1.layerProofs.count {
            let lp1 = proof1.layerProofs[i]
            let lp2 = proof2.layerProofs[i]
            for j in 0..<lp1.sumcheckMsgs.count {
                if frToInt(lp1.sumcheckMsgs[j].s0) != frToInt(lp2.sumcheckMsgs[j].s0) { match = false }
                if frToInt(lp1.sumcheckMsgs[j].s1) != frToInt(lp2.sumcheckMsgs[j].s1) { match = false }
            }
            if frToInt(lp1.claimedVx) != frToInt(lp2.claimedVx) { match = false }
            if frToInt(lp1.claimedVy) != frToInt(lp2.claimedVy) { match = false }
        }
        expect(match, "GKR proofs are deterministic (Fiat-Shamir)")
    }
}
