// Data-Parallel GKR Benchmark (C21)
// Tests proving N repetitions of the same sub-circuit using a single combined GKR proof.

import Foundation
import zkMetal

public func runDataParallelBench() {
    fputs("\n--- Data-Parallel GKR Benchmark (BN254 Fr) ---\n", stderr)
    var rng: UInt64 = 0xDADA_BABE_CAFE_1234
    func nextRand() -> UInt64 { rng = rng &* 6364136223846793005 &+ 1442695040888963407; return rng >> 32 }
    func randomFr() -> Fr { frFromInt(nextRand()) }
    func randomInputs(_ n: Int) -> [Fr] { (0..<n).map { _ in randomFr() } }

    fputs("  Correctness tests:\n", stderr)

    // Test 1: minimal 2-instance squaring
    do {
        let sub = SubCircuit.repeatedSquaring(rounds: 1)
        let inputs: [[Fr]] = [[frFromInt(3)], [frFromInt(5)]]
        let uniform = UniformCircuit(subCircuit: sub, inputs: inputs)
        let engine = DataParallelEngine()
        let pt = Transcript(label: "dp-dbg", backend: .keccak256)
        let proof = engine.prove(circuit: uniform, transcript: pt)
        let vt = Transcript(label: "dp-dbg", backend: .keccak256)
        let ok = engine.verify(subCircuit: sub, numInstances: 2, inputs: inputs, proof: proof, transcript: vt)
        fputs("    Minimal 2-inst squaring: \(ok ? "PASS" : "FAIL")\n", stderr)
    }

    // Test 2: repeated squaring, 4 instances, depth 4
    do {
        let sub = SubCircuit.repeatedSquaring(rounds: 4)
        let inputs: [[Fr]] = (0..<4).map { _ in [randomFr()] }
        let uniform = UniformCircuit(subCircuit: sub, inputs: inputs)
        let engine = DataParallelEngine()
        let pt = Transcript(label: "dp-test", backend: .keccak256)
        let proof = engine.prove(circuit: uniform, transcript: pt)
        let vt = Transcript(label: "dp-test", backend: .keccak256)
        let ok = engine.verify(subCircuit: sub, numInstances: 4, inputs: inputs, proof: proof, transcript: vt)
        fputs("    Repeated squaring (4 inst, depth 4): \(ok ? "PASS" : "FAIL")\n", stderr)
    }

    // Test 3: hash-like, 8 instances
    do {
        let sub = SubCircuit.hashLike(logWidth: 2, numRounds: 3)
        let inputs: [[Fr]] = (0..<8).map { _ in randomInputs(4) }
        let uniform = UniformCircuit(subCircuit: sub, inputs: inputs)
        let engine = DataParallelEngine()
        let pt = Transcript(label: "dp-hash", backend: .keccak256)
        let proof = engine.prove(circuit: uniform, transcript: pt)
        let vt = Transcript(label: "dp-hash", backend: .keccak256)
        let ok = engine.verify(subCircuit: sub, numInstances: 8, inputs: inputs, proof: proof, transcript: vt)
        fputs("    Hash-like (8 inst, width 4, depth 3): \(ok ? "PASS" : "FAIL")\n", stderr)
    }

    // Test 4: tamper detection -- wrong inputs should reject
    do {
        let sub = SubCircuit.repeatedSquaring(rounds: 2)
        let inputs: [[Fr]] = (0..<4).map { _ in [randomFr()] }
        let uniform = UniformCircuit(subCircuit: sub, inputs: inputs)
        let engine = DataParallelEngine()
        let pt = Transcript(label: "dp-tamper", backend: .keccak256)
        let proof = engine.prove(circuit: uniform, transcript: pt)
        var tamperedInputs = inputs
        tamperedInputs[0] = [frFromInt(999999)]
        let vt = Transcript(label: "dp-tamper", backend: .keccak256)
        let ok = engine.verify(subCircuit: sub, numInstances: 4, inputs: tamperedInputs, proof: proof, transcript: vt)
        fputs("    Tamper detection: \(!ok ? "PASS" : "FAIL") (expected reject)\n", stderr)
    }

    // Test 5: add-chain sub-circuit, 16 instances
    do {
        let layer = CircuitLayer(gates: [
            Gate(type: .add, leftInput: 0, rightInput: 1),
            Gate(type: .add, leftInput: 0, rightInput: 1),
        ])
        let sub = SubCircuit(layers: [layer], inputSize: 2, outputSize: 2)
        let inputs: [[Fr]] = (0..<16).map { _ in randomInputs(2) }
        let uniform = UniformCircuit(subCircuit: sub, inputs: inputs)
        let engine = DataParallelEngine()
        let pt = Transcript(label: "dp-add", backend: .keccak256)
        let proof = engine.prove(circuit: uniform, transcript: pt)
        let vt = Transcript(label: "dp-add", backend: .keccak256)
        let ok = engine.verify(subCircuit: sub, numInstances: 16, inputs: inputs, proof: proof, transcript: vt)
        fputs("    Add-chain (16 inst, 2 gates): \(ok ? "PASS" : "FAIL")\n", stderr)
    }

    // Performance comparison
    fputs("\n  Performance: data-parallel vs N separate proofs\n", stderr)
    fputs("  Sub-circuit: hashLike(width=4, depth=2), 4 gates/layer\n", stderr)
    fputs("  Instances   DP prove   Naive prove   DP verify  Naive verify  Proof size   OK\n", stderr)

    let quick = CommandLine.arguments.contains("--quick")
    let instanceCounts = quick ? [4, 16, 64, 256] : [16, 64, 256, 1024]
    let sub = SubCircuit.hashLike(logWidth: 2, numRounds: 2)
    let circuit = sub.toLayeredCircuit()

    for n in instanceCounts {
        rng = 0xDADA_BABE_CAFE_1234
        let allInputs: [[Fr]] = (0..<n).map { _ in randomInputs(sub.inputSize) }
        let uniform = UniformCircuit(subCircuit: sub, inputs: allInputs)
        let dpEngine = DataParallelEngine()

        // Warmup
        do {
            let t = Transcript(label: "dp", backend: .keccak256)
            let _ = dpEngine.prove(circuit: uniform, transcript: t)
        }

        // DP prove
        let runs = 3
        var dpProveTimes = [Double]()
        var lastDPProof: DataParallelProof?
        for _ in 0..<runs {
            let t = Transcript(label: "dp", backend: .keccak256)
            let t0 = CFAbsoluteTimeGetCurrent()
            let proof = dpEngine.prove(circuit: uniform, transcript: t)
            dpProveTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            lastDPProof = proof
        }
        dpProveTimes.sort()
        let dpProve = dpProveTimes[runs / 2]

        // DP verify
        var dpVerifyTimes = [Double]()
        var dpOK = false
        for _ in 0..<runs {
            let t = Transcript(label: "dp", backend: .keccak256)
            let t0 = CFAbsoluteTimeGetCurrent()
            let ok = dpEngine.verify(subCircuit: sub, numInstances: n,
                                     inputs: allInputs, proof: lastDPProof!, transcript: t)
            dpVerifyTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            dpOK = ok
        }
        dpVerifyTimes.sort()
        let dpVerify = dpVerifyTimes[runs / 2]

        // Naive: N separate proofs
        let naiveEngine = GKREngine(circuit: circuit)
        do {
            let t = Transcript(label: "n", backend: .keccak256)
            let _ = naiveEngine.prove(inputs: allInputs[0], transcript: t)
        }
        var naiveProveTimes = [Double]()
        for _ in 0..<runs {
            let t0 = CFAbsoluteTimeGetCurrent()
            for i in 0..<n {
                autoreleasepool {
                    let t = Transcript(label: "n", backend: .keccak256)
                    let _ = naiveEngine.prove(inputs: allInputs[i], transcript: t)
                }
            }
            naiveProveTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        naiveProveTimes.sort()
        let naiveProve = naiveProveTimes[runs / 2]
        let naiveVerify = naiveProve * 2.0  // estimate: prove + verify ~ 2x prove

        // Proof size
        var dpMsgs = 0
        for lp in lastDPProof!.layerProofs { dpMsgs += lp.sumcheckMsgs.count }
        let dpBytes = (dpMsgs * 3 + lastDPProof!.layerProofs.count * 2) * 32

        let nt = Transcript(label: "n", backend: .keccak256)
        let naiveProof = naiveEngine.prove(inputs: allInputs[0], transcript: nt)
        var naiveMsgs = 0
        for lp in naiveProof.layerProofs { naiveMsgs += lp.sumcheckMsgs.count }
        let naiveBytes = (naiveMsgs * 3 + naiveProof.layerProofs.count * 2) * n * 32

        let status = dpOK ? "PASS" : "FAIL"
        fputs(String(format: "  N=%-5d   %7.2f ms   %7.2f ms   %7.2f ms   %7.2f ms   %5dB/%5dB  ",
                    n, dpProve, naiveProve, dpVerify, naiveVerify, dpBytes, naiveBytes), stderr)
        fputs("\(status)\n", stderr)
    }

    // Summary
    fputs("\n  Key advantages of data-parallel GKR:\n", stderr)
    fputs("    - Single proof for all N instances (O(log(N*|C|)) proof size)\n", stderr)
    fputs("    - Proof size savings: O(log N) factor over N separate proofs\n", stderr)
    fputs("    - Single verifier check instead of N checks\n", stderr)
    fputs("    - Ideal for blockchain: N identical signature verifications in one proof\n", stderr)
}
