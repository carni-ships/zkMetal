// WHIR Benchmark — correctness and performance comparison vs FRI
import zkMetal
import Foundation

public func runWHIRBench() {
    fputs("=== WHIR Benchmark ===\n", stderr)
    do {
        var rng: UInt64 = 0xDEAD_BEEF_1234
        func nextRand() -> UInt64 { rng = rng &* 6364136223846793005 &+ 1442695040888963407; return rng >> 32 }
        func randomEvals(_ n: Int) -> [Fr] { (0..<n).map { _ in frFromInt(nextRand()) } }

        fputs("\n--- Correctness (2^10) ---\n", stderr)
        let testEvals = randomEvals(1 << 10)

        // q=4, r=4
        let eng1 = try WHIREngine(numQueries: 4, reductionFactor: 4)
        let p1 = try eng1.prove(evaluations: testEvals)
        let vf1 = eng1.verifyFull(proof: p1, evaluations: testEvals)
        let vs1 = eng1.verify(proof: p1, evaluations: testEvals)
        let vn1 = eng1.verify(proof: p1)
        fputs("  (q=4,r=4): \(p1.numRounds) rounds, \(p1.proofSizeBytes) B"
            + " | full=\(vf1 ? "OK" : "FAIL") succinct=\(vs1 ? "OK" : "FAIL") blind=\(vn1 ? "OK" : "FAIL")\n", stderr)

        // q=4, r=2
        let eng2 = try WHIREngine(numQueries: 4, reductionFactor: 2)
        let p2 = try eng2.prove(evaluations: testEvals)
        let vf2 = eng2.verifyFull(proof: p2, evaluations: testEvals)
        fputs("  (q=4,r=2): \(p2.numRounds) rounds, \(p2.proofSizeBytes) B | full=\(vf2 ? "OK" : "FAIL")\n", stderr)

        // q=2, r=4 (minimal)
        let eng3 = try WHIREngine(numQueries: 2, reductionFactor: 4)
        let p3 = try eng3.prove(evaluations: testEvals)
        let vf3 = eng3.verifyFull(proof: p3, evaluations: testEvals)
        fputs("  (q=2,r=4): \(p3.numRounds) rounds, \(p3.proofSizeBytes) B | full=\(vf3 ? "OK" : "FAIL")\n", stderr)

        // Performance benchmarks
        for logN in [10, 14] {
            let benchN = 1 << logN
            let benchEvals = randomEvals(benchN)

            fputs("\n--- Performance (2^\(logN)) ---\n", stderr)

            let configs: [(String, Int, Int)] = [("q=4,r=4", 4, 4), ("q=2,r=4", 2, 4)]
            for cfg in configs {
                let (qLabel, q, r) = cfg
                let whirEng = try WHIREngine(numQueries: q, reductionFactor: r)
                let _ = try whirEng.prove(evaluations: benchEvals)  // warmup
                // Profile one run at 2^14
                if logN == 14 && q == 4 {
                    whirEng.profileProve = true
                    let _ = try whirEng.prove(evaluations: benchEvals)
                    whirEng.profileProve = false
                }

                var wt = [Double](); var wps = 0; var wnr = 0
                for _ in 0..<5 {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let p = try whirEng.prove(evaluations: benchEvals)
                    wt.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                    wps = p.proofSizeBytes; wnr = p.numRounds
                }
                wt.sort()

                let wp = try whirEng.prove(evaluations: benchEvals)
                let vOk = whirEng.verifyFull(proof: wp, evaluations: benchEvals)

                var wvt = [Double]()
                for _ in 0..<10 {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let _ = whirEng.verifyFull(proof: wp, evaluations: benchEvals)
                    wvt.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                }
                wvt.sort()

                fputs("  WHIR (\(qLabel)): \(wnr) rnds, prove \(String(format: "%.1f", wt[2]))ms,"
                    + " verify \(String(format: "%.1f", wvt[5]))ms,"
                    + " proof \(String(format: "%.1f", Double(wps)/1024))KB"
                    + " [\(vOk ? "OK" : "FAIL")]\n", stderr)
            }

            // FRI comparison
            if logN >= 14 {
                let friEngine = try FRIEngine()
                var betas = [Fr]()
                for i in 0..<logN { betas.append(frFromInt(UInt64(i + 1) * 17)) }
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
                    let depth = max(1, Int(log2(Double(max(2, layer.count)))))
                    fps += 2 * (2 * frSize + depth * frSize)
                }
                fputs("  FRI  (GPU):     \(logN) rnds, prove \(String(format: "%.1f", ft[2]))ms,"
                    + " proof ~\(String(format: "%.1f", Double(fps)/1024))KB\n", stderr)
            }
        }

        fputs("\nDone.\n", stderr)
    } catch {
        fputs("Error: \(error)\n", stderr)
    }
}
