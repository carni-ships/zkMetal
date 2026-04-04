// plonk_bench — Benchmark and correctness test for Plonk prover/verifier
//
// Example circuit: prove knowledge of x such that x^3 + x + 5 = y
// Decomposed into gates:
//   v1 = x * x          (mul)
//   v2 = v1 * x         (mul)
//   v3 = v2 + x         (add)
//   v4 = constant(5)
//   y  = v3 + v4        (add)
//
// Then benchmark at various circuit sizes by repeating/padding.

import Foundation
import zkMetal

func runPlonkBench() {
    fputs("\n--- Plonk Benchmark ---\n", stderr)

    do {
        // --- Correctness test: x^3 + x + 5 = y ---
        fputs("Correctness test: x^3 + x + 5 = y\n", stderr)

        let builder = PlonkCircuitBuilder()

        // Allocate input variable x
        let x = builder.addInput()

        // v1 = x * x
        let v1 = builder.mul(x, x)

        // v2 = v1 * x (need copy constraint: the x here is same variable)
        let v2 = builder.mul(v1, x)

        // v3 = v2 + x
        let v3 = builder.add(v2, x)

        // v4 = 5
        let five = frFromInt(5)
        let v4 = builder.constant(five)

        // y = v3 + v4
        let y = builder.add(v3, v4)

        let circuit = builder.build().padded()
        let n = circuit.numGates
        fputs("  Circuit: \(builder.gates.count) gates, padded to \(n)\n", stderr)

        // Generate test SRS
        let srsSecret: [UInt32] = [0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222,
                                    0x33333333, 0x44444444, 0x55555555, 0x00000001]
        let srsSecretFr = frFromLimbs(srsSecret)
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let generator = PointAffine(x: gx, y: gy)
        let srs = KZGEngine.generateTestSRS(secret: srsSecret, size: n + 3, generator: generator)

        let kzg = try KZGEngine(srs: srs)
        let nttEngine = try NTTEngine()

        // Preprocess
        let t0 = CFAbsoluteTimeGetCurrent()
        let preprocessor = PlonkPreprocessor(kzg: kzg, ntt: nttEngine)
        let setup = try preprocessor.setup(circuit: circuit, srsSecret: srsSecretFr)
        let setupTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        fputs("  Setup: \(String(format: "%.1f", setupTime))ms\n", stderr)

        // Build witness: x = 3, then x^3 + x + 5 = 27 + 3 + 5 = 35
        let xVal = frFromInt(3)
        let v1Val = frMul(xVal, xVal)          // 9
        let v2Val = frMul(v1Val, xVal)          // 27
        let v3Val = frAdd(v2Val, xVal)          // 30
        let v4Val = five                         // 5
        let yVal = frAdd(v3Val, v4Val)          // 35

        // Build full witness array indexed by variable
        let totalVars = builder.nextVariable
        var witness = [Fr](repeating: Fr.zero, count: totalVars)
        witness[x] = xVal
        witness[v1] = v1Val
        witness[v2] = v2Val
        witness[v3] = v3Val
        witness[v4] = v4Val
        witness[y] = yVal

        // For constant gate dummy wires, set to zero (already done)
        // For padded gate wires, also zero

        // Extend witness for padded gates
        let paddedCircuit = circuit
        let maxVar = paddedCircuit.wireAssignments.flatMap { $0 }.max() ?? 0
        if maxVar >= witness.count {
            witness += [Fr](repeating: Fr.zero, count: maxVar - witness.count + 1)
        }

        // Prove
        let prover = PlonkProver(setup: setup, kzg: kzg, ntt: nttEngine)
        let t1 = CFAbsoluteTimeGetCurrent()
        let proof = try prover.prove(witness: witness, circuit: paddedCircuit)
        let proveTime = (CFAbsoluteTimeGetCurrent() - t1) * 1000
        fputs("  Prove: \(String(format: "%.1f", proveTime))ms\n", stderr)

        // Verify
        let verifier = PlonkVerifier(setup: setup, kzg: kzg)
        let t2 = CFAbsoluteTimeGetCurrent()
        let valid = verifier.verify(proof: proof)
        let verifyTime = (CFAbsoluteTimeGetCurrent() - t2) * 1000
        fputs("  Verify: \(String(format: "%.1f", verifyTime))ms [\(valid ? "PASS" : "FAIL")]\n", stderr)

        if !valid {
            fputs("  ERROR: Valid proof failed verification!\n", stderr)
        }

        // --- Benchmark at various sizes ---
        let logSizes = CommandLine.arguments.contains("--quick") ? [4, 6] : [4, 6, 8, 10]
        fputs("\n  Size benchmarks:\n", stderr)
        fputs("  \(String(format: "%6s", "n")) | \(String(format: "%10s", "setup(ms)")) | \(String(format: "%10s", "prove(ms)")) | \(String(format: "%10s", "verify(ms)"))\n", stderr)
        fputs("  " + String(repeating: "-", count: 50) + "\n", stderr)

        for logN in logSizes {
            let size = 1 << logN
            // Build a circuit of `size` gates by repeating the cubic pattern
            let (benchCircuit, benchWitness) = buildBenchCircuit(size: size)

            // SRS for this size
            let benchSRS = KZGEngine.generateTestSRS(secret: srsSecret, size: size + 3, generator: generator)
            let benchKZG = try KZGEngine(srs: benchSRS)

            // Setup
            let benchPreprocessor = PlonkPreprocessor(kzg: benchKZG, ntt: nttEngine)
            let st0 = CFAbsoluteTimeGetCurrent()
            let benchSetup = try benchPreprocessor.setup(circuit: benchCircuit, srsSecret: srsSecretFr)
            let st1 = CFAbsoluteTimeGetCurrent()
            let setupMs = (st1 - st0) * 1000

            let benchProver = PlonkProver(setup: benchSetup, kzg: benchKZG, ntt: nttEngine)

            // Prove (warmup + measure)
            let _ = try benchProver.prove(witness: benchWitness, circuit: benchCircuit)
            let pt0 = CFAbsoluteTimeGetCurrent()
            let benchProof = try benchProver.prove(witness: benchWitness, circuit: benchCircuit)
            let pt1 = CFAbsoluteTimeGetCurrent()
            let proveMs = (pt1 - pt0) * 1000

            // Verify
            let benchVerifier = PlonkVerifier(setup: benchSetup, kzg: benchKZG)
            let vt0 = CFAbsoluteTimeGetCurrent()
            let benchValid = benchVerifier.verify(proof: benchProof)
            let vt1 = CFAbsoluteTimeGetCurrent()
            let verifyMs = (vt1 - vt0) * 1000

            let status = benchValid ? "OK" : "FAIL"
            fputs("  \(String(format: "%6d", size)) | \(String(format: "%10.1f", setupMs)) | \(String(format: "%10.1f", proveMs)) | \(String(format: "%10.1f", verifyMs)) [\(status)]\n", stderr)
        }

    } catch {
        fputs("Plonk bench error: \(error)\n", stderr)
    }
}

/// Build a benchmark circuit of given size filled with add/mul gates
private func buildBenchCircuit(size: Int) -> (PlonkCircuit, [Fr]) {
    let builder = PlonkCircuitBuilder()

    // Start with two inputs
    let x0 = builder.addInput()
    let x1 = builder.addInput()

    var prevA = x0
    var prevB = x1

    // Fill with alternating add/mul gates
    var gateCount = 0
    while gateCount < size - 2 {  // -2 for padding room
        if gateCount % 2 == 0 {
            let out = builder.add(prevA, prevB)
            prevA = prevB
            prevB = out
        } else {
            let out = builder.mul(prevA, prevB)
            prevA = prevB
            prevB = out
        }
        gateCount += 1
    }

    let circuit = builder.build().padded()

    // Build witness
    let totalVars = (circuit.wireAssignments.flatMap { $0 }.max() ?? 0) + 1
    var witness = [Fr](repeating: Fr.zero, count: totalVars)
    witness[x0] = frFromInt(3)
    witness[x1] = frFromInt(7)

    // Evaluate gates to fill witness
    for i in 0..<builder.gates.count {
        let wires = builder.wireAssignments[i]
        let a = witness[wires[0]]
        let b = witness[wires[1]]
        let g = builder.gates[i]

        // Compute expected output: qL*a + qR*b + qO*c + qM*a*b + qC = 0
        // => c = -(qL*a + qR*b + qM*a*b + qC) / qO
        let num = frAdd(frAdd(frMul(g.qL, a), frMul(g.qR, b)),
                        frAdd(frMul(g.qM, frMul(a, b)), g.qC))
        // qO should be -1, so c = num / 1 = num
        // Actually: qO*c = -(qL*a + qR*b + qM*a*b + qC), so c = -num/qO
        let negNum = frSub(Fr.zero, num)
        let c = frMul(negNum, frInverse(g.qO))
        witness[wires[2]] = c
    }

    return (circuit, witness)
}
