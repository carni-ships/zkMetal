// marlin_bench — Benchmark and correctness test for Marlin/AHP verifier
//
// Tests:
//   1. Correctness: generate test proof, verify it passes, verify tampered proof fails
//   2. Single verification timing at various constraint sizes
//   3. Batch verification: 10, 50, 100 proofs

import Foundation
import zkMetal

public func runMarlinBench() {
    fputs("\n--- Marlin/AHP Verifier Benchmark ---\n", stderr)

    do {
        // SRS setup
        let srsSecret: [UInt32] = [0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222,
                                    0x33333333, 0x44444444, 0x55555555, 0x00000001]
        let srsSecretFr = frFromLimbs(srsSecret)
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let generator = PointAffine(x: gx, y: gy)

        // --- Correctness test ---
        fputs("Correctness test...\n", stderr)

        let testSRSSize = 512
        let srs = KZGEngine.generateTestSRS(secret: srsSecret, size: testSRSSize, generator: generator)
        let kzg = try KZGEngine(srs: srs)
        let prover = MarlinTestProver(kzg: kzg)
        let verifier = MarlinVerifier(kzg: kzg)

        let publicInput = [frFromInt(3), frFromInt(7), frFromInt(11)]
        let (vk, proof) = try prover.generateTestProof(
            numConstraints: 8, publicInput: publicInput, srsSecret: srsSecretFr)

        let valid = verifier.verify(vk: vk, publicInput: publicInput, proof: proof)
        fputs("  Valid proof:   \(valid ? "PASS" : "FAIL")\n", stderr)

        // Tampered proof: modify an evaluation
        let tamperedEvals = MarlinEvaluations(
            zABeta: frAdd(proof.evaluations.zABeta, Fr.one),
            zBBeta: proof.evaluations.zBBeta,
            wBeta: proof.evaluations.wBeta,
            tBeta: proof.evaluations.tBeta,
            gGamma: proof.evaluations.gGamma,
            hGamma: proof.evaluations.hGamma,
            rowGamma: proof.evaluations.rowGamma,
            colGamma: proof.evaluations.colGamma,
            valGamma: proof.evaluations.valGamma,
            rowColGamma: proof.evaluations.rowColGamma
        )
        let tamperedProof = MarlinProof(
            wCommit: proof.wCommit, zACommit: proof.zACommit, zBCommit: proof.zBCommit,
            tCommit: proof.tCommit, sumcheckPolyCoeffs: proof.sumcheckPolyCoeffs,
            gCommit: proof.gCommit, hCommit: proof.hCommit,
            evaluations: tamperedEvals, openingProofs: proof.openingProofs
        )
        let rejected = !verifier.verify(vk: vk, publicInput: publicInput, proof: tamperedProof)
        fputs("  Tampered proof: \(rejected ? "PASS (rejected)" : "FAIL (accepted)")\n", stderr)

        // --- Single verification benchmark at various sizes ---
        fputs("\nSingle verification timing:\n", stderr)

        let constraintSizes = [8, 16, 32, 64, 128]
        for numC in constraintSizes {
            let srsNeeded = numC * 4 + 64
            if srsNeeded > testSRSSize { continue }

            let pi = [frFromInt(5), frFromInt(13)]
            let (testVK, testProof) = try prover.generateTestProof(
                numConstraints: numC, publicInput: pi, srsSecret: srsSecretFr)

            // Warmup
            let _ = verifier.verify(vk: testVK, publicInput: pi, proof: testProof)

            // Timed runs
            let runs = 5
            var times = [Double]()
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = verifier.verify(vk: testVK, publicInput: pi, proof: testProof)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[runs / 2]
            fputs(String(format: "  m=%3d constraints: %.2f ms\n", numC, median), stderr)
        }

        // --- Batch verification benchmark ---
        fputs("\nBatch verification:\n", stderr)

        let numC = 16
        let pi = [frFromInt(5), frFromInt(13)]
        let (batchVK, _) = try prover.generateTestProof(
            numConstraints: numC, publicInput: pi, srsSecret: srsSecretFr)

        for batchSize in [10, 50, 100] {
            var batch = [(publicInput: [Fr], proof: MarlinProof)]()
            for i in 0..<batchSize {
                let batchPI = [frFromInt(UInt64(5 + i)), frFromInt(13)]
                let (_, batchProof) = try prover.generateTestProof(
                    numConstraints: numC, publicInput: batchPI, srsSecret: srsSecretFr)
                batch.append((publicInput: batchPI, proof: batchProof))
            }

            // Warmup
            let _ = verifier.batchVerify(vk: batchVK, proofs: batch)

            // Single verify time
            let t0s = CFAbsoluteTimeGetCurrent()
            for (bpi, bp) in batch {
                let _ = verifier.verify(vk: batchVK, publicInput: bpi, proof: bp)
            }
            let singleTotalTime = (CFAbsoluteTimeGetCurrent() - t0s) * 1000

            // Batch verify time
            let runs = 3
            var times = [Double]()
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = verifier.batchVerify(vk: batchVK, proofs: batch)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let batchMedian = times[runs / 2]
            let speedup = singleTotalTime / batchMedian

            fputs(String(format: "  %3d proofs: batch %.1f ms | individual %.1f ms | %.1fx\n",
                        batchSize, batchMedian, singleTotalTime, speedup), stderr)
        }

        fputs("\nDone.\n", stderr)

    } catch {
        fputs("Error: \(error)\n", stderr)
    }
}
