import zkMetal
import Foundation

public func runWHIRBench() {
    fputs("=== WHIR Benchmark ===\n", stderr)
    do {
        fputs("\n--- Correctness ---\n", stderr)
        var rng: UInt64 = 0xDEAD_BEEF_1234
        let testN = 1 << 10
        var testEvals = [Fr](repeating: Fr.zero, count: testN)
        for i in 0..<testN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testEvals[i] = frFromInt(rng >> 32)
        }

        let engine = try WHIREngine(numQueries: 16, reductionFactor: 4)
        let proof = try engine.prove(evaluations: testEvals)
        fputs("  Prove 2^10: \(proof.numRounds) rounds, final=\(proof.finalPoly.count), \(proof.proofSizeBytes) bytes\n", stderr)
        let verified = engine.verifyFull(proof: proof, evaluations: testEvals)
        fputs("  VerifyFull: \(verified ? "PASS" : "FAIL")\n", stderr)

        let engine2 = try WHIREngine(numQueries: 8, reductionFactor: 2)
        let proof2 = try engine2.prove(evaluations: testEvals)
        let v2 = engine2.verifyFull(proof: proof2, evaluations: testEvals)
        fputs("  Verify (q=8, r=2): \(v2 ? "PASS" : "FAIL"), \(proof2.numRounds) rounds, \(proof2.proofSizeBytes) bytes\n", stderr)

        fputs("\n--- Performance (2^14) ---\n", stderr)
        let benchN = 1 << 14
        var benchEvals = [Fr](repeating: Fr.zero, count: benchN)
        for i in 0..<benchN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            benchEvals[i] = frFromInt(rng >> 32)
        }

        // WHIR
        let whirEng = try WHIREngine(numQueries: 32, reductionFactor: 4)
        let _ = try whirEng.prove(evaluations: benchEvals)
        var wt = [Double](); var wps = 0
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let p = try whirEng.prove(evaluations: benchEvals)
            wt.append((CFAbsoluteTimeGetCurrent() - t0) * 1000); wps = p.proofSizeBytes
        }
        wt.sort()
        fputs(String(format: "  WHIR prove: %.2fms, proof %d bytes (%.1f KB)\n", wt[2], wps, Double(wps)/1024), stderr)

        // WHIR verify
        let wp = try whirEng.prove(evaluations: benchEvals)
        var wvt = [Double]()
        for _ in 0..<10 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let _ = whirEng.verifyFull(proof: wp, evaluations: benchEvals)
            wvt.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        wvt.sort()
        fputs(String(format: "  WHIR verify: %.3fms\n", wvt[5]), stderr)

        // FRI
        let friEngine = try FRIEngine()
        var betas = [Fr](); for i in 0..<14 { betas.append(frFromInt(UInt64(i + 1) * 17)) }
        let _ = try friEngine.commitPhase(evals: benchEvals, betas: betas)
        var ft = [Double]()
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let _ = try friEngine.commitPhase(evals: benchEvals, betas: betas)
            ft.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        ft.sort()
        let fc = try friEngine.commitPhase(evals: benchEvals, betas: betas)
        let frSize = MemoryLayout<Fr>.stride
        var fps = fc.roots.count * frSize + frSize
        for layer in fc.layers {
            fps += 2 * (2 * frSize + Int(log2(Double(layer.count))) * frSize)
        }
        fputs(String(format: "  FRI  prove: %.2fms, proof ~%d bytes (%.1f KB)\n", ft[2], fps, Double(fps)/1024), stderr)
        fputs(String(format: "  Ratio: WHIR/FRI prover = %.2fx\n", wt[2] / ft[2]), stderr)

        fputs("\nDone.\n", stderr)
    } catch {
        fputs("Error: \(error)\n", stderr)
    }
}
