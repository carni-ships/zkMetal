import Foundation
import zkMetal

public func runSpartanBench() {
    fputs("\n--- Spartan Transparent SNARK ---\n", stderr)
    do {
        let engine = try SpartanEngine()

        // Correctness: x^2 + x + 5 = y
        fputs("\nCorrectness: x^2+x+5=y\n", stderr)
        let (inst, gen) = SpartanR1CSBuilder.exampleQuadratic()
        let xv = frFromInt(3)
        let (pub, wit) = gen(xv)
        let z = SpartanR1CS.buildZ(publicInputs: pub, witness: wit)
        fputs("  R1CS satisfied: \(inst.isSatisfied(z: z) ? "PASS" : "FAIL")\n", stderr)

        let t0 = CFAbsoluteTimeGetCurrent()
        let pf = try engine.prove(instance: inst, publicInputs: pub, witness: wit)
        let proveMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        let t1 = CFAbsoluteTimeGetCurrent()
        let ok = engine.verify(instance: inst, publicInputs: pub, proof: pf)
        let verifyMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000
        fputs("  Prove: \(String(format: "%.1f", proveMs))ms\n", stderr)
        fputs("  Verify: \(String(format: "%.1f", verifyMs))ms \(ok ? "PASS" : "FAIL")\n", stderr)

        // Soundness: tampered public input should be rejected
        let bad = engine.verify(instance: inst, publicInputs: [frFromInt(999)], proof: pf)
        fputs("  Wrong rejected: \(!bad ? "PASS" : "FAIL")\n", stderr)

        // Multiple x values
        fputs("\n  Multi-value test:\n", stderr)
        var allPass = ok && !bad
        for xVal: UInt64 in [0, 1, 7, 42, 100] {
            let x = frFromInt(xVal)
            let (p, w) = gen(x)
            let proof = try engine.prove(instance: inst, publicInputs: p, witness: w)
            let valid = engine.verify(instance: inst, publicInputs: p, proof: proof)
            fputs("    x=\(xVal): \(valid ? "PASS" : "FAIL")\n", stderr)
            allPass = allPass && valid
        }

        fputs("\n--- Benchmark (synthetic R1CS) ---\n", stderr)
        let logSizes = CommandLine.arguments.contains("--quick") ? [4, 6, 8, 10] : [6, 8, 10, 12, 14]
        for logN in logSizes {
            let n = 1 << logN
            let (si, sp, sw) = SpartanR1CSBuilder.syntheticR1CS(numConstraints: n)
            // Warmup
            let _ = try engine.prove(instance: si, publicInputs: sp, witness: sw)
            let runs = logN <= 10 ? 5 : 3
            var pt = [Double](), vt = [Double]()
            for _ in 0..<runs {
                let ps = CFAbsoluteTimeGetCurrent()
                let p = try engine.prove(instance: si, publicInputs: sp, witness: sw)
                pt.append((CFAbsoluteTimeGetCurrent() - ps) * 1000)
                let vs = CFAbsoluteTimeGetCurrent()
                let _ = engine.verify(instance: si, publicInputs: sp, proof: p)
                vt.append((CFAbsoluteTimeGetCurrent() - vs) * 1000)
            }
            pt.sort(); vt.sort()
            let pms = String(format: "%.1f", pt[runs / 2])
            let vms = String(format: "%.1f", vt[runs / 2])
            fputs("  2^\(logN) (\(n) constraints): prove \(pms)ms, verify \(vms)ms\n", stderr)
        }

        fputs("\n  Transparent: YES (no trusted setup)\n", stderr)
        fputs("  Commitment: Basefold (Poseidon2 Merkle)\n", stderr)
        fputs("  Result: \(allPass ? "ALL PASS" : "SOME FAILED")\n", stderr)
        fputs("  Version: \(SpartanEngine.version.description)\n", stderr)
    } catch {
        fputs("Error: \(error)\n", stderr)
    }
}
