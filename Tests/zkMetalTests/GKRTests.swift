import zkMetal
import Foundation

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

    // Helper: compare two Fr values
    func frEqual(_ a: Fr, _ b: Fr) -> Bool {
        return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
               a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
    }

    // =========================================================================
    // SECTION 1: Circuit evaluation tests
    // =========================================================================

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

    // =========================================================================
    // SECTION 2: MultilinearPoly tests
    // =========================================================================

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

    // =========================================================================
    // SECTION 3: GKR prove/verify basic tests
    // =========================================================================

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

    // =========================================================================
    // SECTION 4: 2-layer multiplication circuit with hand-computed values
    // =========================================================================

    do {
        // Circuit: 4 inputs [a, b, c, d]
        // Layer 0: gate0 = a * b, gate1 = c * d  (2 mul gates)
        // Layer 1: gate0 = (a*b) * (c*d)          (1 mul gate)
        let l0 = CircuitLayer(gates: [
            Gate(type: .mul, leftInput: 0, rightInput: 1),
            Gate(type: .mul, leftInput: 2, rightInput: 3),
        ])
        let l1 = CircuitLayer(gates: [
            Gate(type: .mul, leftInput: 0, rightInput: 1),
        ])
        let circuit = LayeredCircuit(layers: [l0, l1])

        // a=2, b=3, c=5, d=7
        let inputs = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7)]
        let allValues = circuit.evaluate(inputs: inputs)

        // Hand-computed:
        //   layer 0 input = [2, 3, 5, 7]
        //   layer 0 output = [2*3=6, 5*7=35]
        //   layer 1 output = [6*35=210]
        expectEqual(frToInt(allValues[1][0])[0], UInt64(6), "Hand-computed: 2*3 = 6")
        expectEqual(frToInt(allValues[1][1])[0], UInt64(35), "Hand-computed: 5*7 = 35")
        expectEqual(frToInt(allValues[2][0])[0], UInt64(210), "Hand-computed: 6*35 = 210")

        let output = circuit.evaluateOutput(inputs: inputs)
        expectEqual(frToInt(output[0])[0], UInt64(210), "Hand-computed output: 2*3*5*7 = 210")

        // GKR prove/verify on this hand-computed circuit
        let engine = GKREngine(circuit: circuit)
        let pt = Transcript(label: "hand-mul", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)
        let vt = Transcript(label: "hand-mul", backend: .keccak256)
        let ok = engine.verify(inputs: inputs, output: output, proof: proof, transcript: vt)
        expect(ok, "GKR hand-computed 2-layer mul circuit")
    }

    // =========================================================================
    // SECTION 5: Addition-only circuit (degenerate case)
    // =========================================================================

    do {
        // All-add circuit: 4 inputs, reduce to 1 via addition tree
        // Layer 0: gate0 = a+b, gate1 = c+d
        // Layer 1: gate0 = (a+b)+(c+d) = a+b+c+d
        let l0 = CircuitLayer(gates: [
            Gate(type: .add, leftInput: 0, rightInput: 1),
            Gate(type: .add, leftInput: 2, rightInput: 3),
        ])
        let l1 = CircuitLayer(gates: [
            Gate(type: .add, leftInput: 0, rightInput: 1),
        ])
        let circuit = LayeredCircuit(layers: [l0, l1])

        let inputs = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let output = circuit.evaluateOutput(inputs: inputs)
        // 10+20+30+40 = 100
        expectEqual(frToInt(output[0])[0], UInt64(100), "Add-only circuit: 10+20+30+40 = 100")

        let engine = GKREngine(circuit: circuit)
        let pt = Transcript(label: "add-only", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)
        let vt = Transcript(label: "add-only", backend: .keccak256)
        let ok = engine.verify(inputs: inputs, output: output, proof: proof, transcript: vt)
        expect(ok, "GKR add-only circuit prove/verify")
    }

    // =========================================================================
    // SECTION 6: Grand Product GKR
    // =========================================================================

    // Test: grand product of small known values (prover correctness)
    do {
        // Product of [2, 3, 5, 7] = 210
        let values = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7)]
        let pt = Transcript(label: "gp-small", backend: .keccak256)
        let proof = GrandProductEngine.prove(values: values, transcript: pt)
        expectEqual(frToInt(proof.claimedProduct)[0], UInt64(210), "Grand product: 2*3*5*7 = 210")
        expectEqual(proof.layerProofs.count, 2, "Grand product depth = log2(4) = 2")
    }

    // Test: grand product matches naive computation for random values
    do {
        let n = 16
        let values = pseudoRandomInputs(n, seed: 0xDEAD_BEEF)
        var naiveProduct = Fr.one
        for v in values { naiveProduct = frMul(naiveProduct, v) }

        let pt = Transcript(label: "gp-rand", backend: .keccak256)
        let proof = GrandProductEngine.prove(values: values, transcript: pt)
        expect(frEqual(proof.claimedProduct, naiveProduct), "Grand product matches naive for n=16")
        expectEqual(proof.layerProofs.count, 4, "Grand product n=16 has depth=4")
    }

    // Test: grand product with non-power-of-2 size (padding test)
    do {
        let values = [frFromInt(2), frFromInt(3), frFromInt(5)]  // 3 elements, pads to 4
        let pt = Transcript(label: "gp-pad", backend: .keccak256)
        let proof = GrandProductEngine.prove(values: values, transcript: pt)
        // 2*3*5 = 30, padded with 1 -> still 30
        expectEqual(frToInt(proof.claimedProduct)[0], UInt64(30), "Grand product with padding: 2*3*5 = 30")
    }

    // Test: grand product with all ones
    do {
        let values = [Fr](repeating: Fr.one, count: 8)
        let pt = Transcript(label: "gp-ones", backend: .keccak256)
        let proof = GrandProductEngine.prove(values: values, transcript: pt)
        expect(frEqual(proof.claimedProduct, Fr.one), "Grand product of all ones = 1")
    }

    // Test: grand product soundness - tampered values have different product
    do {
        let values = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7)]
        let pt = Transcript(label: "gp-tamper", backend: .keccak256)
        let proof = GrandProductEngine.prove(values: values, transcript: pt)

        // Verify the claimed product would differ if values changed
        let tampered = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(11)]
        var tamperedProduct = Fr.one
        for v in tampered { tamperedProduct = frMul(tamperedProduct, v) }
        expect(!frEqual(proof.claimedProduct, tamperedProduct), "Grand product detects tampered values")
    }

    // Test: batch grand product (prover correctness)
    do {
        let sets: [[Fr]] = [
            [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7)],
            [frFromInt(1), frFromInt(4), frFromInt(6), frFromInt(8)],
        ]
        let pt = Transcript(label: "gp-batch", backend: .keccak256)
        let batchProof = GrandProductEngine.proveBatch(valueSets: sets, transcript: pt)
        expectEqual(batchProof.batchSize, 2, "Batch has 2 products")
        expectEqual(frToInt(batchProof.proofs[0].claimedProduct)[0], UInt64(210), "Batch product 0: 210")
        expectEqual(frToInt(batchProof.proofs[1].claimedProduct)[0], UInt64(192), "Batch product 1: 192")
    }

    // Test: grand product determinism — same input produces same proof
    do {
        let values = [frFromInt(2), frFromInt(3), frFromInt(5), frFromInt(7)]
        let pt1 = Transcript(label: "gp-det", backend: .keccak256)
        let proof1 = GrandProductEngine.prove(values: values, transcript: pt1)
        let pt2 = Transcript(label: "gp-det", backend: .keccak256)
        let proof2 = GrandProductEngine.prove(values: values, transcript: pt2)
        expect(frEqual(proof1.claimedProduct, proof2.claimedProduct), "Grand product proofs are deterministic")
        var match = true
        for i in 0..<proof1.layerProofs.count {
            if !frEqual(proof1.layerProofs[i].claimedVLeft, proof2.layerProofs[i].claimedVLeft) { match = false }
            if !frEqual(proof1.layerProofs[i].claimedVRight, proof2.layerProofs[i].claimedVRight) { match = false }
        }
        expect(match, "Grand product layer proofs are deterministic")
    }

    // =========================================================================
    // SECTION 7: Memory Checking GKR
    // =========================================================================

    // Test: simple write-then-read memory trace
    do {
        let builder = MemoryTraceBuilder()
        builder.write(address: 0, value: frFromInt(42))
        builder.write(address: 1, value: frFromInt(99))
        let v0 = builder.read(address: 0)
        let v1 = builder.read(address: 1)

        expectEqual(frToInt(v0)[0], UInt64(42), "Memory read addr 0 = 42")
        expectEqual(frToInt(v1)[0], UInt64(99), "Memory read addr 1 = 99")

        let trace = builder.build()
        let pt = Transcript(label: "mem-simple", backend: .keccak256)
        let proof = MemoryCheckingProver.prove(trace: trace, transcript: pt)

        // Verify proof structure: augmented trace adds init writes + final reads
        expect(proof.numReads > 0, "Memory proof has reads")
        expect(proof.numWrites > 0, "Memory proof has writes")
        expectEqual(proof.initWrites.count, 2, "Memory proof: 2 init writes (addr 0, 1)")
        expectEqual(proof.finalReads.count, 2, "Memory proof: 2 final reads")
        expect(!proof.grandProductProof.claimedProduct.isZero, "Read product is non-zero")
        expect(!proof.writeGrandProductProof.claimedProduct.isZero, "Write product is non-zero")
    }

    // Test: overwrite and re-read
    do {
        let builder = MemoryTraceBuilder()
        builder.write(address: 0, value: frFromInt(10))
        let v1 = builder.read(address: 0)
        expectEqual(frToInt(v1)[0], UInt64(10), "Memory read after write = 10")

        builder.write(address: 0, value: frFromInt(20))
        let v2 = builder.read(address: 0)
        expectEqual(frToInt(v2)[0], UInt64(20), "Memory read after overwrite = 20")

        let trace = builder.build()
        let pt = Transcript(label: "mem-overwrite", backend: .keccak256)
        let proof = MemoryCheckingProver.prove(trace: trace, transcript: pt)
        // Single address: 1 init write, 1 final read
        expectEqual(proof.initWrites.count, 1, "Overwrite: 1 init write")
        expectEqual(proof.finalReads.count, 1, "Overwrite: 1 final read")
        // Timestamp diffs should all be positive
        var allPositive = true
        for d in proof.timestampDiffs { if d.isZero { allPositive = false } }
        expect(allPositive, "Overwrite: all timestamp diffs positive")
    }

    // Test: multiple addresses interleaved
    do {
        let builder = MemoryTraceBuilder()
        builder.write(address: 0, value: frFromInt(100))
        builder.write(address: 1, value: frFromInt(200))
        builder.write(address: 2, value: frFromInt(300))
        let v2 = builder.read(address: 2)
        let v0 = builder.read(address: 0)
        let v1 = builder.read(address: 1)
        expectEqual(frToInt(v2)[0], UInt64(300), "Interleaved read addr 2")
        expectEqual(frToInt(v0)[0], UInt64(100), "Interleaved read addr 0")
        expectEqual(frToInt(v1)[0], UInt64(200), "Interleaved read addr 1")

        let trace = builder.build()
        let pt = Transcript(label: "mem-interleaved", backend: .keccak256)
        let proof = MemoryCheckingProver.prove(trace: trace, transcript: pt)
        expectEqual(proof.initWrites.count, 3, "Interleaved: 3 init writes")
        expectEqual(proof.finalReads.count, 3, "Interleaved: 3 final reads")
        // gamma should be non-zero (derived from Fiat-Shamir)
        expect(!proof.gamma.isZero, "Interleaved: gamma is non-zero")
    }

    // Test: read from uninitialized address returns zero
    do {
        let builder = MemoryTraceBuilder()
        let v = builder.read(address: 5)
        expect(frEqual(v, Fr.zero), "Uninitialized memory reads as zero")

        builder.write(address: 5, value: frFromInt(77))
        let v2 = builder.read(address: 5)
        expectEqual(frToInt(v2)[0], UInt64(77), "After write, read returns written value")

        let trace = builder.build()
        let pt = Transcript(label: "mem-uninit", backend: .keccak256)
        let proof = MemoryCheckingProver.prove(trace: trace, transcript: pt)
        // The proof should have init writes and final reads for the accessed address
        expectEqual(proof.initWrites.count, 1, "Uninit: 1 init write for address 5")
        expectEqual(proof.initWrites[0].0, UInt64(5), "Uninit: init write is for address 5")
    }

    // Test: memory trace metadata
    do {
        let builder = MemoryTraceBuilder()
        builder.write(address: 0, value: frFromInt(1))
        builder.write(address: 1, value: frFromInt(2))
        builder.write(address: 2, value: frFromInt(3))
        builder.read(address: 0)
        builder.read(address: 1)

        let trace = builder.build()
        expectEqual(trace.ops.count, 5, "Trace has 5 operations")
        expectEqual(trace.numAddresses, 3, "Trace uses 3 addresses")
        expectEqual(trace.reads.count, 2, "Trace has 2 reads")
        expectEqual(trace.writes.count, 3, "Trace has 3 writes")
    }

    // Test: memory proof has positive timestamp diffs
    do {
        let builder = MemoryTraceBuilder()
        builder.write(address: 0, value: frFromInt(1))
        builder.read(address: 0)
        builder.write(address: 0, value: frFromInt(2))
        builder.read(address: 0)

        let trace = builder.build()
        let pt = Transcript(label: "mem-ts", backend: .keccak256)
        let proof = MemoryCheckingProver.prove(trace: trace, transcript: pt)
        // All timestamp diffs should be non-zero (positive)
        var allPositive = true
        for d in proof.timestampDiffs {
            if d.isZero { allPositive = false }
        }
        expect(allPositive, "Memory proof: all timestamp diffs are positive")
    }

    // Test: memory trace builder reset
    do {
        let builder = MemoryTraceBuilder()
        builder.write(address: 0, value: frFromInt(1))
        builder.reset()
        // After reset, reading should return zero (fresh state)
        let v = builder.read(address: 0)
        expect(frEqual(v, Fr.zero), "After reset, memory reads as zero")
        let trace = builder.build()
        expectEqual(trace.ops.count, 1, "After reset, trace has only the new op")
    }

    // =========================================================================
    // SECTION 8: GKR with different circuit widths (2, 4, 8, 16 gates per layer)
    // =========================================================================

    for logWidth in 1...4 {
        let width = 1 << logWidth
        let circuit = LayeredCircuit.repeatedHashCircuit(logWidth: logWidth, depth: 2)
        let inputs = pseudoRandomInputs(width, seed: UInt64(logWidth) &* 12345)
        let output = circuit.evaluateOutput(inputs: inputs)

        let engine = GKREngine(circuit: circuit)
        let pt = Transcript(label: "width-\(width)", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)
        let vt = Transcript(label: "width-\(width)", backend: .keccak256)
        let ok = engine.verify(inputs: inputs, output: output, proof: proof, transcript: vt)
        expect(ok, "GKR width=\(width) (logWidth=\(logWidth)) prove/verify")
    }

    // =========================================================================
    // SECTION 9: GKR rejects tampered witness (modify one gate output)
    // =========================================================================

    // Tampered output: modify one gate output value in the claimed output
    do {
        let circuit = LayeredCircuit.repeatedHashCircuit(logWidth: 3, depth: 3)
        let inputs = pseudoRandomInputs(8, seed: 0xBAD_F00D)
        let output = circuit.evaluateOutput(inputs: inputs)

        let engine = GKREngine(circuit: circuit)
        let pt = Transcript(label: "tamper-out", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)

        // Tamper: flip one output value
        var tamperedOutput = output
        tamperedOutput[0] = frAdd(tamperedOutput[0], Fr.one)
        let vt = Transcript(label: "tamper-out", backend: .keccak256)
        let reject = engine.verify(inputs: inputs, output: tamperedOutput, proof: proof, transcript: vt)
        expect(!reject, "GKR rejects tampered output (single gate flipped)")
    }

    // Tampered input: modify one input element
    do {
        let circuit = LayeredCircuit.repeatedHashCircuit(logWidth: 3, depth: 3)
        let inputs = pseudoRandomInputs(8, seed: 0xBAD_CAFE)
        let output = circuit.evaluateOutput(inputs: inputs)

        let engine = GKREngine(circuit: circuit)
        let pt = Transcript(label: "tamper-in", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)

        // Tamper: modify input[3]
        var tamperedInputs = inputs
        tamperedInputs[3] = frAdd(tamperedInputs[3], frFromInt(1))
        let vt = Transcript(label: "tamper-in", backend: .keccak256)
        let reject = engine.verify(inputs: tamperedInputs, output: output, proof: proof, transcript: vt)
        expect(!reject, "GKR rejects tampered input (single element modified)")
    }

    // Tampered with correct-looking but wrong output (swapped gates)
    do {
        let l0 = CircuitLayer(gates: [
            Gate(type: .add, leftInput: 0, rightInput: 1),
            Gate(type: .mul, leftInput: 0, rightInput: 1),
        ])
        let circuit = LayeredCircuit(layers: [l0])
        let inputs = [frFromInt(4), frFromInt(6)]
        let output = circuit.evaluateOutput(inputs: inputs)
        // output = [10, 24]

        let engine = GKREngine(circuit: circuit)
        let pt = Transcript(label: "tamper-swap", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)

        // Swap outputs: [24, 10] instead of [10, 24]
        let swappedOutput = [output[1], output[0]]
        let vt = Transcript(label: "tamper-swap", backend: .keccak256)
        let reject = engine.verify(inputs: inputs, output: swappedOutput, proof: proof, transcript: vt)
        expect(!reject, "GKR rejects swapped gate outputs")
    }

    // =========================================================================
    // SECTION 10: Performance test - 10-layer circuit with 2^10 gates per layer
    // =========================================================================

    do {
        let logWidth = 10
        let depth = 10
        let width = 1 << logWidth
        let circuit = LayeredCircuit.repeatedHashCircuit(logWidth: logWidth, depth: depth)
        let inputs = pseudoRandomInputs(width, seed: 0xBEEF_CAFE)
        let output = circuit.evaluateOutput(inputs: inputs)

        let engine = GKREngine(circuit: circuit)

        let proveStart = CFAbsoluteTimeGetCurrent()
        let pt = Transcript(label: "perf", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)
        let proveTime = CFAbsoluteTimeGetCurrent() - proveStart

        let verifyStart = CFAbsoluteTimeGetCurrent()
        let vt = Transcript(label: "perf", backend: .keccak256)
        let ok = engine.verify(inputs: inputs, output: output, proof: proof, transcript: vt)
        let verifyTime = CFAbsoluteTimeGetCurrent() - verifyStart

        expect(ok, "GKR perf test (2^10 x 10 layers) prove/verify")
        print(String(format: "  GKR perf: 2^%d gates x %d layers — prove %.2fms, verify %.2fms",
                     logWidth, depth, proveTime * 1000, verifyTime * 1000))
        print(String(format: "  GKR proof size: %d layer proofs, %d sumcheck msgs total",
                     proof.layerProofs.count,
                     proof.layerProofs.reduce(0) { $0 + $1.sumcheckMsgs.count }))
    }

    // =========================================================================
    // SECTION 11: Grand product larger random test
    // =========================================================================

    do {
        let n = 64
        let values = pseudoRandomInputs(n, seed: 0x1234_5678)
        var naiveProduct = Fr.one
        for v in values { naiveProduct = frMul(naiveProduct, v) }

        let pt = Transcript(label: "gp-large", backend: .keccak256)
        let proof = GrandProductEngine.prove(values: values, transcript: pt)
        expect(frEqual(proof.claimedProduct, naiveProduct), "Grand product n=64 matches naive")
        expectEqual(proof.layerProofs.count, 6, "Grand product n=64 has depth=6")
    }

    // =========================================================================
    // SECTION 12: Edge case - power-of-two grand products of small sizes
    // =========================================================================

    // Two elements (smallest valid grand product)
    do {
        let values = [frFromInt(6), frFromInt(7)]
        let pt = Transcript(label: "gp-min", backend: .keccak256)
        let proof = GrandProductEngine.prove(values: values, transcript: pt)
        expectEqual(frToInt(proof.claimedProduct)[0], UInt64(42), "Grand product min: 6*7 = 42")
        expectEqual(proof.layerProofs.count, 1, "Grand product n=2 has depth=1")
    }

    // Four elements
    do {
        let values = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let pt = Transcript(label: "gp-four", backend: .keccak256)
        let proof = GrandProductEngine.prove(values: values, transcript: pt)
        expectEqual(frToInt(proof.claimedProduct)[0], UInt64(24), "Grand product: 1*2*3*4 = 24")
        expectEqual(proof.layerProofs.count, 2, "Grand product n=4 has depth=2")
    }

    // =========================================================================
    // SECTION 13: Grand product with identity elements
    // =========================================================================

    // Product where some elements are 1 (multiplicative identity)
    do {
        let values = [frFromInt(1), frFromInt(13), frFromInt(1), frFromInt(17)]
        let pt = Transcript(label: "gp-ident", backend: .keccak256)
        let proof = GrandProductEngine.prove(values: values, transcript: pt)
        // 1 * 13 * 1 * 17 = 221
        expectEqual(frToInt(proof.claimedProduct)[0], UInt64(221), "Grand product with identities: 221")
    }

    // =========================================================================
    // SECTION 14: Memory checking - larger trace with many operations
    // =========================================================================

    do {
        let builder = MemoryTraceBuilder()
        // Write sequential values to 8 addresses
        for i in 0..<8 {
            builder.write(address: UInt64(i), value: frFromInt(UInt64(i * 10 + 1)))
        }
        // Read all back
        for i in 0..<8 {
            let v = builder.read(address: UInt64(i))
            expectEqual(frToInt(v)[0], UInt64(i * 10 + 1), "Memory bulk read addr \(i)")
        }
        // Overwrite even addresses
        for i in stride(from: 0, to: 8, by: 2) {
            builder.write(address: UInt64(i), value: frFromInt(UInt64(i * 100)))
        }
        // Read all again, verify updates
        for i in 0..<8 {
            let v = builder.read(address: UInt64(i))
            if i % 2 == 0 {
                expectEqual(frToInt(v)[0], UInt64(i * 100), "Memory updated addr \(i)")
            } else {
                expectEqual(frToInt(v)[0], UInt64(i * 10 + 1), "Memory unchanged addr \(i)")
            }
        }

        let trace = builder.build()
        expectEqual(trace.ops.count, 28, "Larger trace: 8 writes + 8 reads + 4 overwrites + 8 reads = 28")
        expectEqual(trace.numAddresses, 8, "Larger trace: 8 distinct addresses")

        let pt = Transcript(label: "mem-large", backend: .keccak256)
        let proof = MemoryCheckingProver.prove(trace: trace, transcript: pt)
        expectEqual(proof.initWrites.count, 8, "Larger trace: 8 init writes")
        expectEqual(proof.finalReads.count, 8, "Larger trace: 8 final reads")
        // Verify all timestamp diffs are positive
        var allPositive = true
        for d in proof.timestampDiffs { if d.isZero { allPositive = false } }
        expect(allPositive, "Larger trace: all timestamp diffs positive")
    }

    // =========================================================================
    // SECTION 15: GKR with mixed add/mul patterns at various widths
    // =========================================================================

    // Width=2: minimal circuit
    do {
        let layer = CircuitLayer(gates: [
            Gate(type: .add, leftInput: 0, rightInput: 1),
            Gate(type: .mul, leftInput: 0, rightInput: 1),
        ])
        let circuit = LayeredCircuit(layers: [layer, layer])
        let inputs = [frFromInt(2), frFromInt(3)]
        let output = circuit.evaluateOutput(inputs: inputs)
        // Layer 0: [2+3=5, 2*3=6]
        // Layer 1: [5+6=11, 5*6=30]
        expectEqual(frToInt(output[0])[0], UInt64(11), "Width=2 mixed: add output = 11")
        expectEqual(frToInt(output[1])[0], UInt64(30), "Width=2 mixed: mul output = 30")

        let engine = GKREngine(circuit: circuit)
        let pt = Transcript(label: "mix-2", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)
        let vt = Transcript(label: "mix-2", backend: .keccak256)
        let ok = engine.verify(inputs: inputs, output: output, proof: proof, transcript: vt)
        expect(ok, "GKR width=2 mixed add/mul 2-layer")
    }

    // =========================================================================
    // SECTION 16: Grand product performance
    // =========================================================================

    do {
        let n = 256
        let values = pseudoRandomInputs(n, seed: 0xFACE_B00C)

        var naiveProduct = Fr.one
        for v in values { naiveProduct = frMul(naiveProduct, v) }

        let proveStart = CFAbsoluteTimeGetCurrent()
        let pt = Transcript(label: "gp-perf", backend: .keccak256)
        let proof = GrandProductEngine.prove(values: values, transcript: pt)
        let proveTime = CFAbsoluteTimeGetCurrent() - proveStart

        expect(frEqual(proof.claimedProduct, naiveProduct), "Grand product perf n=256 correct")
        print(String(format: "  Grand product perf: n=%d — prove %.2fms", n, proveTime * 1000))
    }
}
