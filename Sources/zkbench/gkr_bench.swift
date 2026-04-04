// GKR Benchmark — Prove and verify layered arithmetic circuits
// Tests: hash-like circuits at various depths and widths, inner product circuits
// Key property: GKR avoids NTT entirely — purely algebraic sumcheck-based protocol

import Foundation
import zkMetal

public func runGKRBench() {
    fputs("\n--- GKR Benchmark (BN254 Fr) ---\n", stderr)

    // Tiny correctness test first
    do {
        // 2-gate circuit: gate 0 = add(in0, in1), gate 1 = mul(in0, in1)
        let layer = CircuitLayer(gates: [
            Gate(type: .add, leftInput: 0, rightInput: 1),
            Gate(type: .mul, leftInput: 0, rightInput: 1),
        ])
        let circuit = LayeredCircuit(layers: [layer])
        let a = frFromInt(3)
        let b = frFromInt(5)
        let inputs = [a, b]
        let output = circuit.evaluateOutput(inputs: inputs)
        // output[0] = 3+5=8, output[1] = 3*5=15
        let out0 = frToInt(output[0])
        let out1 = frToInt(output[1])
        fputs("  Tiny test: add=\(out0[0]) mul=\(out1[0])\n", stderr)

        let engine = GKREngine(circuit: circuit)
        let pt = Transcript(label: "tiny", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)
        fputs("  Proof layers: \(proof.layerProofs.count)\n", stderr)
        fputs("  SC msgs: \(proof.layerProofs[0].sumcheckMsgs.count)\n", stderr)

        let vt = Transcript(label: "tiny", backend: .keccak256)
        let ok = engine.verify(inputs: inputs, output: output, proof: proof, transcript: vt)
        fputs("  Tiny test: \(ok ? "PASS" : "FAIL")\n", stderr)

        // Debug: check sum of table vs claim
        let nOut = circuit.outputVars(layer: 0)
        let nIn = circuit.inputVars(layer: 0)
        fputs("  nOut=\(nOut) nIn=\(nIn)\n", stderr)

        // Compute claim manually
        let outputMLE = MultilinearPoly(numVars: nOut, values: output)
        let debugT = Transcript(label: "tiny", backend: .keccak256)
        for v in output { debugT.absorb(v) }
        debugT.absorbLabel("gkr-init")
        let r = debugT.squeezeN(nOut)
        let claim = outputMLE.evaluate(at: r)
        fputs("  claim = \(frToInt(claim))\n", stderr)

        // Compute table sum
        let addMLE = circuit.addMLEForLayer(0)
        let mulMLE = circuit.mulMLEForLayer(0)
        var addF = addMLE
        for i in 0..<nOut { addF = addF.fixVariable(r[i]) }
        var mulF = mulMLE
        for i in 0..<nOut { mulF = mulF.fixVariable(r[i]) }

        let inSize = 1 << nIn
        let tableSize = 1 << (2 * nIn)
        let prevMLE = MultilinearPoly(numVars: nIn, values: inputs)
        var tableSum = Fr.zero
        for idx in 0..<tableSize {
            let xIdx = idx >> nIn
            let yIdx = idx & (inSize - 1)
            let vx = prevMLE.evals[xIdx]
            let vy = prevMLE.evals[yIdx]
            let aVal = addF.evals[idx]
            let mVal = mulF.evals[idx]
            let g = frAdd(frMul(aVal, frAdd(vx, vy)), frMul(mVal, frMul(vx, vy)))
            tableSum = frAdd(tableSum, g)
        }
        fputs("  table sum = \(frToInt(tableSum))\n", stderr)
    }

    // Generate random field inputs
    func randomInputs(_ n: Int) -> [Fr] {
        var rng: UInt64 = 0xCAFE_BABE_DEAD_BEEF
        return (0..<n).map { _ in
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return frFromInt(rng >> 32)
        }
    }

    // Test configurations: (logWidth, depth)
    let configs: [(Int, Int)] = [
        (4, 4),     // 16 gates wide, 4 layers — small sanity check
        (6, 8),     // 64 gates wide, 8 layers
        (8, 8),     // 256 gates wide, 8 layers
        (8, 16),    // 256 gates wide, 16 layers
        (10, 8),    // 1024 gates wide, 8 layers
        (10, 16),   // 1024 gates wide, 16 layers
    ]

    let quick = CommandLine.arguments.contains("--quick")
    let activeConfigs = quick ? Array(configs.prefix(3)) : configs

    fputs("  Hash-like circuits (half add, half mul per layer):\n", stderr)
    fputs("  Width        Depth    Prove(ms)    Verify(ms)   OK?\n", stderr)

    for (logW, depth) in activeConfigs {
        let width = 1 << logW
        let circuit = LayeredCircuit.repeatedHashCircuit(logWidth: logW, depth: depth)
        let inputs = randomInputs(width)

        let engine = GKREngine(circuit: circuit)

        // Correctness check
        let output = circuit.evaluateOutput(inputs: inputs)
        let proveTranscript = Transcript(label: "gkr-bench", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: proveTranscript)
        let verifyTranscript = Transcript(label: "gkr-bench", backend: .keccak256)
        let valid = engine.verify(inputs: inputs, output: output, proof: proof, transcript: verifyTranscript)

        // Benchmark prove
        let proveRuns = 3
        var proveTimes = [Double]()
        for _ in 0..<proveRuns {
            let t = Transcript(label: "gkr-bench", backend: .keccak256)
            let t0 = CFAbsoluteTimeGetCurrent()
            let _ = engine.prove(inputs: inputs, transcript: t)
            proveTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        proveTimes.sort()
        let proveMedian = proveTimes[proveRuns / 2]

        // Benchmark verify
        let verifyRuns = 3
        var verifyTimes = [Double]()
        for _ in 0..<verifyRuns {
            let t = Transcript(label: "gkr-bench", backend: .keccak256)
            let t0 = CFAbsoluteTimeGetCurrent()
            let _ = engine.verify(inputs: inputs, output: output, proof: proof, transcript: t)
            verifyTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        verifyTimes.sort()
        let verifyMedian = verifyTimes[verifyRuns / 2]

        let status = valid ? "PASS" : "FAIL"
        fputs(String(format: "  2^%-4d d=%-4d  %8.2f ms   %8.2f ms   \(status)\n",
                    logW, depth, proveMedian, verifyMedian), stderr)
    }

    // Inner product circuit benchmark
    fputs("\n  Inner product circuits (mul layer + add tree):\n", stderr)
    fputs("  Size         Depth    Prove(ms)    Verify(ms)   OK?\n", stderr)

    let ipSizes = quick ? [4, 6] : [4, 6, 8, 10]
    for logN in ipSizes {
        let n = 1 << logN
        let circuit = LayeredCircuit.innerProductCircuit(size: n)
        // Inner product takes 2*n inputs (pairs of values)
        let inputs = randomInputs(2 * n)

        let engine = GKREngine(circuit: circuit)
        let output = circuit.evaluateOutput(inputs: inputs)

        let pt = Transcript(label: "gkr-ip", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)
        let vt = Transcript(label: "gkr-ip", backend: .keccak256)
        let valid = engine.verify(inputs: inputs, output: output, proof: proof, transcript: vt)

        // Benchmark
        let runs = 3
        var proveTimes = [Double]()
        var verifyTimes = [Double]()
        for _ in 0..<runs {
            let t1 = Transcript(label: "gkr-ip", backend: .keccak256)
            let t0 = CFAbsoluteTimeGetCurrent()
            let _ = engine.prove(inputs: inputs, transcript: t1)
            proveTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)

            let t2 = Transcript(label: "gkr-ip", backend: .keccak256)
            let v0 = CFAbsoluteTimeGetCurrent()
            let _ = engine.verify(inputs: inputs, output: output, proof: proof, transcript: t2)
            verifyTimes.append((CFAbsoluteTimeGetCurrent() - v0) * 1000)
        }
        proveTimes.sort()
        verifyTimes.sort()

        let status = valid ? "PASS" : "FAIL"
        fputs(String(format: "  2^%-4d d=%-4d  %8.2f ms   %8.2f ms   \(status)\n",
                    logN, circuit.depth, proveTimes[runs / 2], verifyTimes[runs / 2]), stderr)
    }

    // Compare with direct evaluation (no NTT needed)
    fputs("\n  GKR key property: no NTT/FFT required (pure algebraic sumcheck)\n", stderr)
    let testLogW = 8
    let testDepth = 8
    let testCircuit = LayeredCircuit.repeatedHashCircuit(logWidth: testLogW, depth: testDepth)
    let testInputs = randomInputs(1 << testLogW)
    let testEngine = GKREngine(circuit: testCircuit)

    // Direct circuit evaluation time
    let directT0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<100 {
        let _ = testCircuit.evaluateOutput(inputs: testInputs)
    }
    let directTime = (CFAbsoluteTimeGetCurrent() - directT0) * 1000 / 100

    // GKR prove time (includes evaluation)
    let gkrT = Transcript(label: "gkr-cmp", backend: .keccak256)
    let gkrT0 = CFAbsoluteTimeGetCurrent()
    let _ = testEngine.prove(inputs: testInputs, transcript: gkrT)
    let gkrTime = (CFAbsoluteTimeGetCurrent() - gkrT0) * 1000

    fputs(String(format: "  Direct eval (2^%d, depth %d): %.3f ms\n", testLogW, testDepth, directTime), stderr)
    fputs(String(format: "  GKR prove (same circuit):     %.3f ms\n", gkrTime), stderr)
    fputs(String(format: "  Proof overhead:               %.1fx\n", gkrTime / directTime), stderr)

    // Proof size
    let sampleT = Transcript(label: "gkr-size", backend: .keccak256)
    let sampleProof = testEngine.prove(inputs: testInputs, transcript: sampleT)
    var totalMsgs = 0
    for lp in sampleProof.layerProofs {
        totalMsgs += lp.sumcheckMsgs.count
    }
    let proofFieldElements = totalMsgs * 3 + sampleProof.layerProofs.count * 2
    let proofBytes = proofFieldElements * 32
    fputs(String(format: "  Proof size: %d field elements (%d bytes)\n", proofFieldElements, proofBytes), stderr)
}
