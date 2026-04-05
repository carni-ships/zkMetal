// STIR Benchmark — correctness and performance comparison vs FRI
import zkMetal
import Foundation

public func runSTIRBench() {
    fputs("=== STIR Benchmark ===\n", stderr)
    do {
        var rng: UInt64 = 0xDEAD_BEEF_5A1D
        func nextRand() -> UInt64 { rng = rng &* 6364136223846793005 &+ 1442695040888963407; return rng >> 32 }
        func randomEvals(_ n: Int) -> [Fr] { (0..<n).map { _ in frFromInt(nextRand()) } }

        // --- Correctness tests ---
        fputs("\n--- Correctness (2^10) ---\n", stderr)
        let testEvals = randomEvals(1 << 10)

        // Domain shift round-trip test
        do {
            let eng = try STIRProver(numQueries: 4, reductionFactor: 4)
            let alpha = frFromInt(42)
            let shifted = try eng.domainShift(evals: testEvals, alpha: alpha)
            let cpuShifted = STIRProver.cpuDomainShift(evals: testEvals, alpha: alpha)
            var shiftOk = shifted.count == cpuShifted.count
            if shiftOk {
                for i in 0..<shifted.count {
                    if frToInt(shifted[i]) != frToInt(cpuShifted[i]) {
                        shiftOk = false
                        break
                    }
                }
            }
            fputs("  Domain shift:  \(shiftOk ? "OK" : "FAIL")\n", stderr)
        }

        // q=4, r=4
        do {
            let eng = try STIRProver(numQueries: 4, reductionFactor: 4)
            let p = try eng.prove(evaluations: testEvals)
            let vf = eng.verifyFull(proof: p, evaluations: testEvals)
            let vs = eng.verify(proof: p, evaluations: testEvals)
            let vn = eng.verify(proof: p)
            fputs("  (q=4,r=4): \(p.numRounds) rounds, \(p.proofSizeBytes) B"
                + " | full=\(vf ? "OK" : "FAIL") succinct=\(vs ? "OK" : "FAIL") blind=\(vn ? "OK" : "FAIL")\n", stderr)
        }

        // q=4, r=2
        do {
            let eng = try STIRProver(numQueries: 4, reductionFactor: 2)
            let p = try eng.prove(evaluations: testEvals)
            let vf = eng.verifyFull(proof: p, evaluations: testEvals)
            fputs("  (q=4,r=2): \(p.numRounds) rounds, \(p.proofSizeBytes) B | full=\(vf ? "OK" : "FAIL")\n", stderr)
        }

        // q=2, r=4 (minimal)
        do {
            let eng = try STIRProver(numQueries: 2, reductionFactor: 4)
            let p = try eng.prove(evaluations: testEvals)
            let vf = eng.verifyFull(proof: p, evaluations: testEvals)
            fputs("  (q=2,r=4): \(p.numRounds) rounds, \(p.proofSizeBytes) B | full=\(vf ? "OK" : "FAIL")\n", stderr)
        }

        // Soundness analysis
        fputs("\n--- Soundness comparison (128-bit security, rate=1/4) ---\n", stderr)
        let friQ = STIRProver.queriesNeeded(securityBits: 128, rate: 0.25, useSTIR: false)
        let stirQ = STIRProver.queriesNeeded(securityBits: 128, rate: 0.25, useSTIR: true)
        fputs("  FRI queries needed:  \(friQ)\n", stderr)
        fputs("  STIR queries needed: \(stirQ) (\(String(format: "%.0f", (1.0 - Double(stirQ)/Double(friQ)) * 100))%% fewer)\n", stderr)

        // --- Performance benchmarks ---
        for logN in [10, 14, 18] {
            let benchN = 1 << logN
            let benchEvals = randomEvals(benchN)

            fputs("\n--- Performance (2^\(logN) = \(benchN)) ---\n", stderr)

            let configs: [(String, Int, Int)] = [("q=4,r=4", 4, 4), ("q=2,r=4", 2, 4)]
            for cfg in configs {
                let (qLabel, q, r) = cfg
                let stirEng = try STIRProver(numQueries: q, reductionFactor: r)
                let _ = try stirEng.prove(evaluations: benchEvals)  // warmup

                var pt = [Double](); var ps = 0; var pnr = 0
                for _ in 0..<5 {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let p = try stirEng.prove(evaluations: benchEvals)
                    pt.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                    ps = p.proofSizeBytes; pnr = p.numRounds
                }
                pt.sort()

                let prf = try stirEng.prove(evaluations: benchEvals)
                let vOk = stirEng.verifyFull(proof: prf, evaluations: benchEvals)

                var vt = [Double]()
                for _ in 0..<10 {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let _ = stirEng.verify(proof: prf)
                    vt.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                }
                vt.sort()

                fputs("  STIR (\(qLabel)): \(pnr) rnds, prove \(String(format: "%.1f", pt[2]))ms,"
                    + " verify \(String(format: "%.3f", vt[5]))ms,"
                    + " proof \(String(format: "%.1f", Double(ps)/1024))KB"
                    + " [\(vOk ? "OK" : "FAIL")]\n", stderr)
            }

            // WHIR comparison (same reduction factor, same queries)
            do {
                let whirEng = try WHIREngine(numQueries: 4, reductionFactor: 4)
                let _ = try whirEng.prove(evaluations: benchEvals)  // warmup

                var wt = [Double](); var wps = 0; var wnr = 0
                for _ in 0..<5 {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let p = try whirEng.prove(evaluations: benchEvals)
                    wt.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                    wps = p.proofSizeBytes; wnr = p.numRounds
                }
                wt.sort()

                fputs("  WHIR (q=4,r=4): \(wnr) rnds, prove \(String(format: "%.1f", wt[2]))ms,"
                    + " proof \(String(format: "%.1f", Double(wps)/1024))KB\n", stderr)
            }

            // FRI comparison (GPU fold-by-4 default, only for larger sizes)
            if logN >= 14 {
                do {
                    let friEngine = try FRIEngine()
                    var betas = [Fr]()
                    let numBetas = friEngine.defaultFoldMode.betaCount(logN: logN)
                    for i in 0..<numBetas { betas.append(frFromInt(UInt64(i + 1) * 17)) }
                    let _ = try friEngine.commit(evals: benchEvals, betas: betas)  // warmup

                    var ft = [Double]()
                    for _ in 0..<5 {
                        let t0 = CFAbsoluteTimeGetCurrent()
                        let _ = try friEngine.commit(evals: benchEvals, betas: betas)
                        ft.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                    }
                    ft.sort()

                    let fc = try friEngine.commit(evals: benchEvals, betas: betas)
                    let frSize = MemoryLayout<Fr>.stride
                    var fps = fc.roots.count * frSize + frSize
                    for layer in fc.layers {
                        let depth = max(1, Int(log2(Double(max(2, layer.count)))))
                        fps += 2 * (2 * frSize + depth * frSize)
                    }
                    fputs("  FRI  (GPU \(friEngine.defaultFoldMode)):  \(numBetas) rnds, prove \(String(format: "%.1f", ft[2]))ms,"
                        + " proof ~\(String(format: "%.1f", Double(fps)/1024))KB\n", stderr)
                }
            }
        }

        fputs("\nDone.\n", stderr)
    } catch {
        fputs("Error: \(error)\n", stderr)
    }
}
