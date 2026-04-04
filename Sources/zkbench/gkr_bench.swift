// GKR Benchmark

import Foundation
import zkMetal

public func runGKRBench() {
    fputs("\n--- GKR Benchmark (BN254 Fr) ---\n", stderr)

    func randomInputs(_ n: Int) -> [Fr] {
        var rng: UInt64 = 0xCAFE_BABE_DEAD_BEEF
        return (0..<n).map { _ in
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return frFromInt(rng >> 32)
        }
    }

    // 1-layer test
    do {
        let layer = CircuitLayer(gates: [
            Gate(type: .add, leftInput: 0, rightInput: 1),
            Gate(type: .mul, leftInput: 0, rightInput: 1),
        ])
        let circuit = LayeredCircuit(layers: [layer])
        let inputs = [frFromInt(3), frFromInt(5)]
        let output = circuit.evaluateOutput(inputs: inputs)
        let engine = GKREngine(circuit: circuit)
        let pt = Transcript(label: "tiny", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)
        let vt = Transcript(label: "tiny", backend: .keccak256)
        let ok = engine.verify(inputs: inputs, output: output, proof: proof, transcript: vt)
        fputs("  1-layer: \(ok ? "PASS" : "FAIL")\n", stderr)
    }

    // 2-layer test
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
        let pt = Transcript(label: "2l", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)
        let vt = Transcript(label: "2l", backend: .keccak256)
        let ok = engine.verify(inputs: inputs, output: output, proof: proof, transcript: vt)
        fputs("  2-layer: \(ok ? "PASS" : "FAIL")\n", stderr)
    }

    let configs: [(Int, Int)] = [(4, 4), (6, 8), (8, 8), (8, 16), (10, 8), (10, 16)]
    let quick = CommandLine.arguments.contains("--quick")
    let active = quick ? Array(configs.prefix(3)) : configs

    fputs("\n  Hash circuits:\n", stderr)
    for (logW, depth) in active {
        let circuit = LayeredCircuit.repeatedHashCircuit(logWidth: logW, depth: depth)
        let inputs = randomInputs(1 << logW)
        let engine = GKREngine(circuit: circuit)
        let output = circuit.evaluateOutput(inputs: inputs)
        let pt = Transcript(label: "g", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)
        let vt = Transcript(label: "g", backend: .keccak256)
        let valid = engine.verify(inputs: inputs, output: output, proof: proof, transcript: vt)

        var pTimes = [Double]()
        for _ in 0..<3 {
            let t = Transcript(label: "g", backend: .keccak256)
            let t0 = CFAbsoluteTimeGetCurrent()
            let _ = engine.prove(inputs: inputs, transcript: t)
            pTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        pTimes.sort()
        var vTimes = [Double]()
        for _ in 0..<3 {
            let t = Transcript(label: "g", backend: .keccak256)
            let t0 = CFAbsoluteTimeGetCurrent()
            let _ = engine.verify(inputs: inputs, output: output, proof: proof, transcript: t)
            vTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        vTimes.sort()
        let st = valid ? "PASS" : "FAIL"
        fputs(String(format: "  2^%-4d d=%-4d %8.2f ms  %8.2f ms  \(st)\n",
                    logW, depth, pTimes[1], vTimes[1]), stderr)
    }

    // Inner product
    fputs("\n  Inner product:\n", stderr)
    for logN in (quick ? [4, 6] : [4, 6, 8]) {
        let n = 1 << logN
        let circuit = LayeredCircuit.innerProductCircuit(size: n)
        let inputs = randomInputs(2 * n)
        let engine = GKREngine(circuit: circuit)
        let output = circuit.evaluateOutput(inputs: inputs)
        let pt = Transcript(label: "i", backend: .keccak256)
        let proof = engine.prove(inputs: inputs, transcript: pt)
        let vt = Transcript(label: "i", backend: .keccak256)
        let valid = engine.verify(inputs: inputs, output: output, proof: proof, transcript: vt)
        let st = valid ? "PASS" : "FAIL"
        fputs("  IP 2^\(logN): \(st)\n", stderr)
    }

    fputs("\n  No NTT required - pure algebraic sumcheck\n", stderr)
}
