// spartan_bench — Benchmark and correctness test for Spartan transparent SNARK

import Foundation
import zkMetal

public func runSpartanBench() {
    fputs("\n--- Spartan Transparent SNARK Benchmark ---\n", stderr)

    do {
        let engine = try SpartanEngine()

        // --- Correctness test: x^2 + x + 5 = y ---
        fputs("\nCorrectness test: x^2 + x + 5 = y\n", stderr)

        let (instance, witnessGen) = R1CSBuilder.exampleQuadratic()
        let xVal = frFromInt(3)
        let (pubInputs, witness) = witnessGen(xVal)

        let z = R1CSInstance.buildZ(publicInputs: pubInputs, witness: witness)
        let satisfied = instance.isSatisfied(z: z)
        fputs("  R1CS satisfied: \(satisfied ? "PASS" : "FAIL")\n", stderr)

        let t0 = CFAbsoluteTimeGetCurrent()
        let proof = try engine.prove(instance: instance, publicInputs: pubInputs, witness: witness)
        let proveTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        fputs("  Prove: \(String(format: "%.1f", proveTime))ms\n", stderr)

        let t1 = CFAbsoluteTimeGetCurrent()
        let verified = engine.verify(instance: instance, publicInputs: pubInputs, proof: proof)
        let verifyTime = (CFAbsoluteTimeGetCurrent() - t1) * 1000
        fputs("  Verify: \(String(format: "%.1f", verifyTime))ms (\(verified ? "PASS" : "FAIL"))\n", stderr)

        // Wrong input should fail
        let wrongPub = [frFromInt(999)]
        let wrongVerified = engine.verify(instance: instance, publicInputs: wrongPub, proof: proof)
        fputs("  Wrong input rejected: \(!wrongVerified ? "PASS" : "FAIL")\n", stderr)

        // --- Benchmark at various sizes ---
        fputs("\n--- Spartan Prove/Verify Benchmark ---\n", stderr)
        fputs(String(format: "  %-12s %10s %10s %10s\n", "Constraints", "Prove(ms)", "Verify(ms)", "Rounds"), stderr)

        let sizes = CommandLine.arguments.contains("--quick") ? [6, 8, 10] : [6, 8, 10, 12, 14]

        for logN in sizes {
            let numConstraints = 1 << logN
            let (synInstance, synPub, synWit) = R1CSBuilder.syntheticR1CS(numConstraints: numConstraints)

            // Warmup
            let _ = try engine.prove(instance: synInstance, publicInputs: synPub, witness: synWit)

            let runs = logN <= 10 ? 5 : 3
            var proveTimes = [Double]()
            var verifyTimes = [Double]()
            var lastProof: SpartanProof!

            for _ in 0..<runs {
                let ps = CFAbsoluteTimeGetCurrent()
                let p = try engine.prove(instance: synInstance, publicInputs: synPub, witness: synWit)
                proveTimes.append((CFAbsoluteTimeGetCurrent() - ps) * 1000)
                lastProof = p

                let vs = CFAbsoluteTimeGetCurrent()
                let _ = engine.verify(instance: synInstance, publicInputs: synPub, proof: p)
                verifyTimes.append((CFAbsoluteTimeGetCurrent() - vs) * 1000)
            }

            proveTimes.sort()
            verifyTimes.sort()
            let medianProve = proveTimes[runs / 2]
            let medianVerify = verifyTimes[runs / 2]
            let numRounds = lastProof.sumcheckRounds.count

            fputs(String(format: "  2^%-9d %9.1fms %9.1fms %10d\n",
                         logN, medianProve, medianVerify, numRounds), stderr)
        }

        // --- Proof metadata ---
        fputs("\n--- Proof Structure (2^10 constraints) ---\n", stderr)
        let (synInst, synPub2, synWit2) = R1CSBuilder.syntheticR1CS(numConstraints: 1 << 10)
        let testProof = try engine.prove(instance: synInst, publicInputs: synPub2, witness: synWit2)
        fputs("  Sumcheck rounds: \(testProof.sumcheckRounds.count)\n", stderr)
        fputs("  Opening proof layers: \(testProof.openingProof.roots.count)\n", stderr)
        fputs("  Opening proof queries: \(testProof.openingProof.queryProofs.count)\n", stderr)

        let frSize = 32
        let scBytes = testProof.sumcheckRounds.count * 3 * frSize
        let overhead = 5 * frSize
        fputs("  Approx proof size (excl. PCS): \(overhead + scBytes) bytes\n", stderr)
        fputs("  Properties: transparent (no trusted setup)\n", stderr)
        fputs("\n  Version: \(SpartanEngine.version.description)\n", stderr)

    } catch {
        fputs("Error: \(error)\n", stderr)
    }
}
