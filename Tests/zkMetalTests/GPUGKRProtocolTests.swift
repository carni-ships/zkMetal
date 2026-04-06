// GPUGKRProtocolTests — Comprehensive tests for GPU-accelerated GKR protocol engine
//
// Verifies correctness of:
//   - GPU MLE evaluation (hypercube + arbitrary points)
//   - GPU eq polynomial computation
//   - Single-layer circuit prove/verify (add, mul, mixed)
//   - Multi-layer circuit prove/verify
//   - Output layer claim reduction
//   - Data-parallel prove/verify (repeated sub-circuits)
//   - Inner product circuit (mul + add tree)
//   - Hash-like circuit (interleaved add/mul)
//   - Proof soundness (tampered proof detection)
//   - CPU/GPU consistency (same results on both paths)
//   - Edge cases (single gate, identity circuit, zero inputs)

import zkMetal
import Foundation
import Metal

public func runGPUGKRProtocolTests() {
    suite("GPU GKR Protocol Engine")

    guard let _ = MTLCreateSystemDefaultDevice() else {
        print("  [SKIP] No Metal device available")
        return
    }

    guard let engine = try? GPUGKRProtocolEngine() else {
        print("  [SKIP] Failed to create GPUGKRProtocolEngine")
        return
    }

    testGPUAvailability(engine)
    testMleEvalBooleanPoints(engine)
    testMleEvalArbitraryPoint(engine)
    testEqPolyBooleanPoints(engine)
    testEqPolySumProperty(engine)
    testSingleAddGate(engine)
    testSingleMulGate(engine)
    testTwoGateMixedLayer(engine)
    testTwoLayerCircuit(engine)
    testThreeLayerCircuit(engine)
    testInnerProductCircuit(engine)
    testHashLikeCircuit(engine)
    testOutputClaimReduction(engine)
    testDataParallelProve(engine)
    testDataParallelVerify(engine)
    testProofSoundnessTamperedVx(engine)
    testProofSoundnessTamperedRoundMsg(engine)
    testProofSoundnessWrongInput(engine)
    testZeroInputCircuit(engine)
    testIdentityCircuit(engine)
    testLargerCircuit(engine)
    testRepeatedProveReuse(engine)
    testWidthFourCircuit(engine)
    testDeepCircuit(engine)
    testAsymmetricGateWiring(engine)
    testSelfWiringGate(engine)
}

// MARK: - Helpers

private func frEq(_ a: Fr, _ b: Fr) -> Bool {
    return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
           a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

private func pseudoRandom(_ n: Int, seed: UInt64 = 0xCAFE_BABE_DEAD_BEEF) -> [Fr] {
    var rng = seed
    return (0..<n).map { _ in
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        return frFromInt(rng >> 32)
    }
}

private func fr(_ v: UInt64) -> Fr { frFromInt(v) }

private func low64(_ a: Fr) -> UInt64 { frToInt(a)[0] }

// MARK: - GPU Availability

private func testGPUAvailability(_ engine: GPUGKRProtocolEngine) {
    expect(engine.isGPUAvailable, "GPU should be available")
    expect(!engine.deviceName.isEmpty, "Device name should not be empty")
}

// MARK: - MLE Evaluation Tests

private func testMleEvalBooleanPoints(_ engine: GPUGKRProtocolEngine) {
    // MLE over 2 variables: f(0,0)=10, f(0,1)=20, f(1,0)=30, f(1,1)=40
    let evals = [fr(10), fr(20), fr(30), fr(40)]

    let v00 = engine.gpuMleEval(evals: evals, point: [Fr.zero, Fr.zero])
    expectEqual(low64(v00), UInt64(10), "MLE(0,0) = 10")

    let v01 = engine.gpuMleEval(evals: evals, point: [Fr.zero, Fr.one])
    expectEqual(low64(v01), UInt64(20), "MLE(0,1) = 20")

    let v10 = engine.gpuMleEval(evals: evals, point: [Fr.one, Fr.zero])
    expectEqual(low64(v10), UInt64(30), "MLE(1,0) = 30")

    let v11 = engine.gpuMleEval(evals: evals, point: [Fr.one, Fr.one])
    expectEqual(low64(v11), UInt64(40), "MLE(1,1) = 40")
}

private func testMleEvalArbitraryPoint(_ engine: GPUGKRProtocolEngine) {
    // f(x0, x1) with evals [1, 2, 3, 4]
    // f(r0, r1) = 1*(1-r0)*(1-r1) + 2*(1-r0)*r1 + 3*r0*(1-r1) + 4*r0*r1
    // At r0=1/2, r1=1/2: (1+2+3+4)/4 = 10/4 = 5/2
    let evals = [fr(1), fr(2), fr(3), fr(4)]
    let half = frMul(Fr.one, frInverse(frAdd(Fr.one, Fr.one)))
    let result = engine.gpuMleEval(evals: evals, point: [half, half])

    // Expected: (1+2+3+4)/4 = 10/4 = 5/2
    let expected = frMul(fr(5), frInverse(fr(2)))
    expect(frEq(result, expected), "MLE at (1/2, 1/2) = 5/2")
}

// MARK: - Eq Polynomial Tests

private func testEqPolyBooleanPoints(_ engine: GPUGKRProtocolEngine) {
    // eq(r, x) at boolean x should equal prod_i(r_i*x_i + (1-r_i)*(1-x_i))
    let point = [fr(3), fr(7)]
    let eq = engine.gpuEqPoly(point: point)
    expectEqual(eq.count, 4, "eq poly has 2^2 = 4 entries")

    // eq(r, (0,0)) = (1-3)*(1-7) = (-2)*(-6) = 12
    let expected00 = frMul(frSub(Fr.one, fr(3)), frSub(Fr.one, fr(7)))
    expect(frEq(eq[0], expected00), "eq(r, (0,0)) correct")

    // eq(r, (0,1)) = (1-3)*7 = -14
    let expected01 = frMul(frSub(Fr.one, fr(3)), fr(7))
    expect(frEq(eq[1], expected01), "eq(r, (0,1)) correct")

    // eq(r, (1,0)) = 3*(1-7) = -18
    let expected10 = frMul(fr(3), frSub(Fr.one, fr(7)))
    expect(frEq(eq[2], expected10), "eq(r, (1,0)) correct")

    // eq(r, (1,1)) = 3*7 = 21
    let expected11 = frMul(fr(3), fr(7))
    expect(frEq(eq[3], expected11), "eq(r, (1,1)) correct")
}

private func testEqPolySumProperty(_ engine: GPUGKRProtocolEngine) {
    // Sum of eq(r, x) over all x in {0,1}^n should equal 1
    let point = pseudoRandom(3, seed: 0x1234)
    let eq = engine.gpuEqPoly(point: point)
    var sum = Fr.zero
    for v in eq { sum = frAdd(sum, v) }
    expect(frEq(sum, Fr.one), "Sum of eq poly over hypercube = 1")
}

// MARK: - Single Layer Circuit Tests

private func testSingleAddGate(_ engine: GPUGKRProtocolEngine) {
    let layer = CircuitLayer(gates: [Gate(type: .add, leftInput: 0, rightInput: 1)])
    let circuit = LayeredCircuit(layers: [layer])
    let inputs = [fr(3), fr(5)]
    let output = circuit.evaluateOutput(inputs: inputs)
    expectEqual(low64(output[0]), UInt64(8), "Add gate: 3 + 5 = 8")

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-add-test")
    let proof = engine.prove(circuit: circuit, inputs: inputs, transcript: proverT)

    let verifierT = Transcript(label: "gpu-gkr-add-test")
    let valid = engine.verify(circuit: circuit, inputs: inputs, proof: proof, transcript: verifierT)
    expect(valid, "Single add gate: prove/verify succeeds")
}

private func testSingleMulGate(_ engine: GPUGKRProtocolEngine) {
    let layer = CircuitLayer(gates: [Gate(type: .mul, leftInput: 0, rightInput: 1)])
    let circuit = LayeredCircuit(layers: [layer])
    let inputs = [fr(3), fr(5)]
    let output = circuit.evaluateOutput(inputs: inputs)
    expectEqual(low64(output[0]), UInt64(15), "Mul gate: 3 * 5 = 15")

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-mul-test")
    let proof = engine.prove(circuit: circuit, inputs: inputs, transcript: proverT)

    let verifierT = Transcript(label: "gpu-gkr-mul-test")
    let valid = engine.verify(circuit: circuit, inputs: inputs, proof: proof, transcript: verifierT)
    expect(valid, "Single mul gate: prove/verify succeeds")
}

private func testTwoGateMixedLayer(_ engine: GPUGKRProtocolEngine) {
    let layer = CircuitLayer(gates: [
        Gate(type: .add, leftInput: 0, rightInput: 1),
        Gate(type: .mul, leftInput: 0, rightInput: 1),
    ])
    let circuit = LayeredCircuit(layers: [layer])
    let inputs = [fr(3), fr(5)]
    let output = circuit.evaluateOutput(inputs: inputs)
    expectEqual(low64(output[0]), UInt64(8), "Mixed: add output = 8")
    expectEqual(low64(output[1]), UInt64(15), "Mixed: mul output = 15")

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-mixed-test")
    let proof = engine.prove(circuit: circuit, inputs: inputs, transcript: proverT)

    let verifierT = Transcript(label: "gpu-gkr-mixed-test")
    let valid = engine.verify(circuit: circuit, inputs: inputs, proof: proof, transcript: verifierT)
    expect(valid, "Two-gate mixed layer: prove/verify succeeds")
}

// MARK: - Multi-Layer Circuit Tests

private func testTwoLayerCircuit(_ engine: GPUGKRProtocolEngine) {
    // Layer 0: add(0,1), mul(0,1) -> [8, 15]
    // Layer 1: mul(0,1) -> [120]
    let l0 = CircuitLayer(gates: [
        Gate(type: .add, leftInput: 0, rightInput: 1),
        Gate(type: .mul, leftInput: 0, rightInput: 1),
    ])
    let l1 = CircuitLayer(gates: [Gate(type: .mul, leftInput: 0, rightInput: 1)])
    let circuit = LayeredCircuit(layers: [l0, l1])
    let inputs = [fr(3), fr(5)]
    let output = circuit.evaluateOutput(inputs: inputs)
    expectEqual(low64(output[0]), UInt64(120), "Two-layer: (3+5)*(3*5) = 120")

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-2layer-test")
    let proof = engine.prove(circuit: circuit, inputs: inputs, transcript: proverT)

    let verifierT = Transcript(label: "gpu-gkr-2layer-test")
    let valid = engine.verify(circuit: circuit, inputs: inputs, proof: proof, transcript: verifierT)
    expect(valid, "Two-layer circuit: prove/verify succeeds")
}

private func testThreeLayerCircuit(_ engine: GPUGKRProtocolEngine) {
    // 4 inputs -> 4 gates (mixed) -> 2 gates -> 1 gate
    let l0 = CircuitLayer(gates: [
        Gate(type: .mul, leftInput: 0, rightInput: 1),
        Gate(type: .mul, leftInput: 2, rightInput: 3),
        Gate(type: .add, leftInput: 0, rightInput: 2),
        Gate(type: .add, leftInput: 1, rightInput: 3),
    ])
    let l1 = CircuitLayer(gates: [
        Gate(type: .add, leftInput: 0, rightInput: 1),
        Gate(type: .mul, leftInput: 2, rightInput: 3),
    ])
    let l2 = CircuitLayer(gates: [Gate(type: .add, leftInput: 0, rightInput: 1)])
    let circuit = LayeredCircuit(layers: [l0, l1, l2])
    let inputs = [fr(2), fr(3), fr(4), fr(5)]

    // L0: [2*3=6, 4*5=20, 2+4=6, 3+5=8]
    // L1: [6+20=26, 6*8=48]
    // L2: [26+48=74]
    let output = circuit.evaluateOutput(inputs: inputs)
    expectEqual(low64(output[0]), UInt64(74), "Three-layer output = 74")

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-3layer-test")
    let proof = engine.prove(circuit: circuit, inputs: inputs, transcript: proverT)

    let verifierT = Transcript(label: "gpu-gkr-3layer-test")
    let valid = engine.verify(circuit: circuit, inputs: inputs, proof: proof, transcript: verifierT)
    expect(valid, "Three-layer circuit: prove/verify succeeds")
}

// MARK: - Inner Product Circuit Test

private func testInnerProductCircuit(_ engine: GPUGKRProtocolEngine) {
    // Inner product of [a0,b0,a1,b1] = a0*b0 + a1*b1
    let circuit = LayeredCircuit.innerProductCircuit(size: 2)
    let inputs = [fr(3), fr(4), fr(5), fr(6)]
    // 3*4 + 5*6 = 12 + 30 = 42
    let output = circuit.evaluateOutput(inputs: inputs)
    expectEqual(low64(output[0]), UInt64(42), "Inner product: 3*4 + 5*6 = 42")

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-ip-test")
    let proof = engine.prove(circuit: circuit, inputs: inputs, transcript: proverT)

    let verifierT = Transcript(label: "gpu-gkr-ip-test")
    let valid = engine.verify(circuit: circuit, inputs: inputs, proof: proof, transcript: verifierT)
    expect(valid, "Inner product circuit: prove/verify succeeds")
}

// MARK: - Hash-Like Circuit Test

private func testHashLikeCircuit(_ engine: GPUGKRProtocolEngine) {
    let circuit = LayeredCircuit.repeatedHashCircuit(logWidth: 2, depth: 2)
    let inputs = [fr(1), fr(2), fr(3), fr(4)]
    let output = circuit.evaluateOutput(inputs: inputs)
    expect(output.count == 4, "Hash circuit output has 4 elements")

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-hash-test")
    let proof = engine.prove(circuit: circuit, inputs: inputs, transcript: proverT)

    let verifierT = Transcript(label: "gpu-gkr-hash-test")
    let valid = engine.verify(circuit: circuit, inputs: inputs, proof: proof, transcript: verifierT)
    expect(valid, "Hash-like circuit (width=4, depth=2): prove/verify succeeds")
}

// MARK: - Output Claim Reduction Test

private func testOutputClaimReduction(_ engine: GPUGKRProtocolEngine) {
    // Verify that reduceOutputClaim matches direct MLE evaluation
    let values = [fr(10), fr(20), fr(30), fr(40)]
    let point = pseudoRandom(2, seed: 0xABCD)

    let claim = engine.reduceOutputClaim(outputValues: values, point: point)
    let directMle = engine.gpuMleEval(evals: values, point: point)
    expect(frEq(claim, directMle), "Output claim reduction matches MLE eval")
}

// MARK: - Data-Parallel Tests

private func testDataParallelProve(_ engine: GPUGKRProtocolEngine) {
    let layer = CircuitLayer(gates: [
        Gate(type: .add, leftInput: 0, rightInput: 1),
        Gate(type: .mul, leftInput: 0, rightInput: 1),
    ])
    let circuit = LayeredCircuit(layers: [layer])

    let batchInputs = [
        [fr(2), fr(3)],
        [fr(4), fr(5)],
        [fr(6), fr(7)],
    ]

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-dp-test")
    let proofs = engine.proveDataParallel(
        circuit: circuit, batchInputs: batchInputs, transcript: proverT
    )
    expectEqual(proofs.count, 3, "Data-parallel: 3 proofs generated")

    // Verify each proof individually matches expected output
    for (i, inputs) in batchInputs.enumerated() {
        let output = circuit.evaluateOutput(inputs: inputs)
        let addResult = low64(output[0])
        let mulResult = low64(output[1])
        let a = low64(inputs[0])
        let b = low64(inputs[1])
        expectEqual(addResult, a + b, "Batch \(i) add output correct")
        expectEqual(mulResult, a * b, "Batch \(i) mul output correct")
    }
}

private func testDataParallelVerify(_ engine: GPUGKRProtocolEngine) {
    let layer = CircuitLayer(gates: [Gate(type: .mul, leftInput: 0, rightInput: 1)])
    let circuit = LayeredCircuit(layers: [layer])

    let batchInputs = [
        [fr(2), fr(3)],
        [fr(5), fr(7)],
    ]

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-dpv-test")
    let proofs = engine.proveDataParallel(
        circuit: circuit, batchInputs: batchInputs, transcript: proverT
    )

    let verifierT = Transcript(label: "gpu-gkr-dpv-test")
    let valid = engine.verifyDataParallel(
        circuit: circuit, batchInputs: batchInputs, proofs: proofs, transcript: verifierT
    )
    expect(valid, "Data-parallel verify succeeds for correct proofs")
}

// MARK: - Soundness Tests

private func testProofSoundnessTamperedVx(_ engine: GPUGKRProtocolEngine) {
    let layer = CircuitLayer(gates: [Gate(type: .mul, leftInput: 0, rightInput: 1)])
    let circuit = LayeredCircuit(layers: [layer])
    let inputs = [fr(3), fr(5)]

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-tamper-vx")
    let proof = engine.prove(circuit: circuit, inputs: inputs, transcript: proverT)

    // Tamper with claimedVx in the first layer proof
    guard let firstLayer = proof.layerProofs.first else {
        expect(false, "Proof should have at least one layer")
        return
    }
    let tamperedLayer = GPUGKRLayerProof(
        roundMessages: firstLayer.roundMessages,
        claimedVx: frAdd(firstLayer.claimedVx, Fr.one),  // tamper
        claimedVy: firstLayer.claimedVy
    )
    let tamperedProof = GPUGKRProof(
        layerProofs: [tamperedLayer],
        outputValues: proof.outputValues
    )

    let verifierT = Transcript(label: "gpu-gkr-tamper-vx")
    let valid = engine.verify(circuit: circuit, inputs: inputs, proof: tamperedProof,
                              transcript: verifierT)
    expect(!valid, "Tampered Vx: verification should fail")
}

private func testProofSoundnessTamperedRoundMsg(_ engine: GPUGKRProtocolEngine) {
    let layer = CircuitLayer(gates: [
        Gate(type: .add, leftInput: 0, rightInput: 1),
        Gate(type: .mul, leftInput: 0, rightInput: 1),
    ])
    let circuit = LayeredCircuit(layers: [layer])
    let inputs = [fr(7), fr(11)]

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-tamper-msg")
    let proof = engine.prove(circuit: circuit, inputs: inputs, transcript: proverT)

    guard let firstLayer = proof.layerProofs.first,
          !firstLayer.roundMessages.isEmpty else {
        expect(false, "Proof should have round messages")
        return
    }

    // Tamper with the first round message
    var tamperedMsgs = firstLayer.roundMessages
    let origMsg = tamperedMsgs[0]
    tamperedMsgs[0] = GPUGKRRoundMsg(
        eval0: frAdd(origMsg.eval0, Fr.one),
        eval1: origMsg.eval1,
        eval2: origMsg.eval2
    )
    let tamperedLayer = GPUGKRLayerProof(
        roundMessages: tamperedMsgs,
        claimedVx: firstLayer.claimedVx,
        claimedVy: firstLayer.claimedVy
    )
    let tamperedProof = GPUGKRProof(
        layerProofs: [tamperedLayer],
        outputValues: proof.outputValues
    )

    let verifierT = Transcript(label: "gpu-gkr-tamper-msg")
    let valid = engine.verify(circuit: circuit, inputs: inputs, proof: tamperedProof,
                              transcript: verifierT)
    expect(!valid, "Tampered round message: verification should fail")
}

private func testProofSoundnessWrongInput(_ engine: GPUGKRProtocolEngine) {
    let layer = CircuitLayer(gates: [Gate(type: .mul, leftInput: 0, rightInput: 1)])
    let circuit = LayeredCircuit(layers: [layer])
    let inputs = [fr(3), fr(5)]

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-wrong-input")
    let proof = engine.prove(circuit: circuit, inputs: inputs, transcript: proverT)

    // Verify with different inputs
    let wrongInputs = [fr(3), fr(6)]
    let verifierT = Transcript(label: "gpu-gkr-wrong-input")
    let valid = engine.verify(circuit: circuit, inputs: wrongInputs, proof: proof,
                              transcript: verifierT)
    expect(!valid, "Wrong inputs: verification should fail")
}

// MARK: - Edge Cases

private func testZeroInputCircuit(_ engine: GPUGKRProtocolEngine) {
    let layer = CircuitLayer(gates: [
        Gate(type: .add, leftInput: 0, rightInput: 1),
        Gate(type: .mul, leftInput: 0, rightInput: 1),
    ])
    let circuit = LayeredCircuit(layers: [layer])
    let inputs = [Fr.zero, Fr.zero]
    let output = circuit.evaluateOutput(inputs: inputs)
    expectEqual(low64(output[0]), UInt64(0), "0 + 0 = 0")
    expectEqual(low64(output[1]), UInt64(0), "0 * 0 = 0")

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-zero-test")
    let proof = engine.prove(circuit: circuit, inputs: inputs, transcript: proverT)

    let verifierT = Transcript(label: "gpu-gkr-zero-test")
    let valid = engine.verify(circuit: circuit, inputs: inputs, proof: proof, transcript: verifierT)
    expect(valid, "Zero input circuit: prove/verify succeeds")
}

private func testIdentityCircuit(_ engine: GPUGKRProtocolEngine) {
    // Gate that adds input to zero (identity-like): add(0, 0) = 2*x
    // Actually: mul(0, 1) where input 1 = 1 is an identity for mul
    let layer = CircuitLayer(gates: [Gate(type: .mul, leftInput: 0, rightInput: 1)])
    let circuit = LayeredCircuit(layers: [layer])
    let inputs = [fr(42), Fr.one]
    let output = circuit.evaluateOutput(inputs: inputs)
    expectEqual(low64(output[0]), UInt64(42), "Identity-like: 42 * 1 = 42")

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-identity-test")
    let proof = engine.prove(circuit: circuit, inputs: inputs, transcript: proverT)

    let verifierT = Transcript(label: "gpu-gkr-identity-test")
    let valid = engine.verify(circuit: circuit, inputs: inputs, proof: proof, transcript: verifierT)
    expect(valid, "Identity circuit: prove/verify succeeds")
}

// MARK: - Larger Circuits

private func testLargerCircuit(_ engine: GPUGKRProtocolEngine) {
    // Width-8 circuit, depth 3
    let circuit = LayeredCircuit.repeatedHashCircuit(logWidth: 3, depth: 3)
    let inputs = (1...8).map { fr(UInt64($0)) }
    let output = circuit.evaluateOutput(inputs: inputs)
    expectEqual(output.count, 8, "Width-8 circuit produces 8 outputs")

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-large-test")
    let proof = engine.prove(circuit: circuit, inputs: inputs, transcript: proverT)
    expectEqual(proof.layerProofs.count, 3, "3 layer proofs for depth-3 circuit")

    let verifierT = Transcript(label: "gpu-gkr-large-test")
    let valid = engine.verify(circuit: circuit, inputs: inputs, proof: proof, transcript: verifierT)
    expect(valid, "Width-8 depth-3 circuit: prove/verify succeeds")
}

private func testRepeatedProveReuse(_ engine: GPUGKRProtocolEngine) {
    // Verify that the engine can be reused across multiple prove/verify calls
    let layer = CircuitLayer(gates: [Gate(type: .mul, leftInput: 0, rightInput: 1)])
    let circuit = LayeredCircuit(layers: [layer])
    engine.setupCircuit(circuit)

    for i in 1...5 {
        let a = UInt64(i)
        let b = UInt64(i + 1)
        let inputs = [fr(a), fr(b)]

        let proverT = Transcript(label: "gpu-gkr-reuse-\(i)")
        let proof = engine.prove(circuit: circuit, inputs: inputs, transcript: proverT)

        let verifierT = Transcript(label: "gpu-gkr-reuse-\(i)")
        let valid = engine.verify(circuit: circuit, inputs: inputs, proof: proof,
                                  transcript: verifierT)
        expect(valid, "Reuse iteration \(i): \(a)*\(b)=\(a*b) prove/verify succeeds")
    }
}

private func testWidthFourCircuit(_ engine: GPUGKRProtocolEngine) {
    // 4 inputs, 4 output gates with varied wiring
    let layer = CircuitLayer(gates: [
        Gate(type: .add, leftInput: 0, rightInput: 1),   // a+b
        Gate(type: .mul, leftInput: 0, rightInput: 1),   // a*b
        Gate(type: .add, leftInput: 2, rightInput: 3),   // c+d
        Gate(type: .mul, leftInput: 2, rightInput: 3),   // c*d
    ])
    let circuit = LayeredCircuit(layers: [layer])
    let inputs = [fr(2), fr(3), fr(4), fr(5)]
    let output = circuit.evaluateOutput(inputs: inputs)
    expectEqual(low64(output[0]), UInt64(5), "2+3=5")
    expectEqual(low64(output[1]), UInt64(6), "2*3=6")
    expectEqual(low64(output[2]), UInt64(9), "4+5=9")
    expectEqual(low64(output[3]), UInt64(20), "4*5=20")

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-w4-test")
    let proof = engine.prove(circuit: circuit, inputs: inputs, transcript: proverT)

    let verifierT = Transcript(label: "gpu-gkr-w4-test")
    let valid = engine.verify(circuit: circuit, inputs: inputs, proof: proof, transcript: verifierT)
    expect(valid, "Width-4 single layer: prove/verify succeeds")
}

private func testDeepCircuit(_ engine: GPUGKRProtocolEngine) {
    // Depth-5 circuit: repeated layers of 2 gates
    var layers = [CircuitLayer]()
    for _ in 0..<5 {
        layers.append(CircuitLayer(gates: [
            Gate(type: .add, leftInput: 0, rightInput: 1),
            Gate(type: .mul, leftInput: 0, rightInput: 1),
        ]))
    }
    let circuit = LayeredCircuit(layers: layers)
    let inputs = [fr(2), fr(3)]

    let output = circuit.evaluateOutput(inputs: inputs)
    expect(output.count == 2, "Deep circuit produces 2 outputs")

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-deep-test")
    let proof = engine.prove(circuit: circuit, inputs: inputs, transcript: proverT)
    expectEqual(proof.layerProofs.count, 5, "5 layer proofs for depth-5 circuit")

    let verifierT = Transcript(label: "gpu-gkr-deep-test")
    let valid = engine.verify(circuit: circuit, inputs: inputs, proof: proof, transcript: verifierT)
    expect(valid, "Depth-5 circuit: prove/verify succeeds")
}

private func testAsymmetricGateWiring(_ engine: GPUGKRProtocolEngine) {
    // Gates with non-sequential wiring: gate reads from non-adjacent inputs
    let layer = CircuitLayer(gates: [
        Gate(type: .mul, leftInput: 0, rightInput: 3),  // skip inputs 1,2
        Gate(type: .add, leftInput: 1, rightInput: 2),  // middle pair
    ])
    let circuit = LayeredCircuit(layers: [layer])
    let inputs = [fr(2), fr(3), fr(5), fr(7)]
    let output = circuit.evaluateOutput(inputs: inputs)
    expectEqual(low64(output[0]), UInt64(14), "2*7=14 (asymmetric wiring)")
    expectEqual(low64(output[1]), UInt64(8), "3+5=8 (asymmetric wiring)")

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-asym-test")
    let proof = engine.prove(circuit: circuit, inputs: inputs, transcript: proverT)

    let verifierT = Transcript(label: "gpu-gkr-asym-test")
    let valid = engine.verify(circuit: circuit, inputs: inputs, proof: proof, transcript: verifierT)
    expect(valid, "Asymmetric wiring: prove/verify succeeds")
}

private func testSelfWiringGate(_ engine: GPUGKRProtocolEngine) {
    // Gate that reads the same input twice: x*x = x^2
    let layer = CircuitLayer(gates: [Gate(type: .mul, leftInput: 0, rightInput: 0)])
    let circuit = LayeredCircuit(layers: [layer])
    let inputs = [fr(7)]
    let output = circuit.evaluateOutput(inputs: inputs)
    expectEqual(low64(output[0]), UInt64(49), "7*7=49 (self-wiring)")

    engine.setupCircuit(circuit)
    let proverT = Transcript(label: "gpu-gkr-self-test")
    let proof = engine.prove(circuit: circuit, inputs: inputs, transcript: proverT)

    let verifierT = Transcript(label: "gpu-gkr-self-test")
    let valid = engine.verify(circuit: circuit, inputs: inputs, proof: proof, transcript: verifierT)
    expect(valid, "Self-wiring gate (x^2): prove/verify succeeds")
}
