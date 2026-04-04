// Batch Verification Benchmark
// Compares: individual verification (N scalar checks) vs batch verification (random LC)
// Tests correctness: batch detects a single invalid proof among N-1 valid ones
// Benchmarks: batch vs individual at N=4, 16, 64, 256
// Tests ProofAggregator with mixed KZG proofs and adaptive CPU/GPU path selection

import Foundation
import zkMetal

public func runBatchVerifyBench() {
    fputs("=== Batch KZG Verification Benchmark ===\n", stderr)

    do {
        // Setup: generate SRS
        let generator = PointAffine(x: fpFromInt(1), y: fpFromInt(2))
        let secret: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]
        let srsSize = 1024
        let srs = KZGEngine.generateTestSRS(secret: secret, size: srsSize, generator: generator)
        let srsSecret = frFromLimbs(secret)
        let kzg = try KZGEngine(srs: srs)
        let batchVerifier = try BatchVerifier(msmEngine: kzg.msmEngine)

        // Generate random polynomials and their KZG proofs
        let quick = CommandLine.arguments.contains("--quick")
        let batchSizes = quick ? [4, 16, 64] : [4, 16, 64, 256]
        let maxN = batchSizes.last!

        fputs("Generating \(maxN) KZG proofs...\n", stderr)
        var rng: UInt64 = 0xCAFE_BABE_DEAD_BEEF

        var allItems = [VerificationItem]()
        allItems.reserveCapacity(maxN)

        let proofGenT0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<maxN {
            // Random polynomial of degree 15
            let degree = 16
            var coeffs = [Fr]()
            coeffs.reserveCapacity(degree)
            for _ in 0..<degree {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                coeffs.append(frFromInt(rng >> 32))
            }

            // Random evaluation point
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let z = frFromInt(rng >> 32)

            // Commit and open
            let commitment = try kzg.commit(coeffs)
            let proof = try kzg.open(coeffs, at: z)

            allItems.append(VerificationItem(
                commitment: commitment,
                point: z,
                value: proof.evaluation,
                proof: proof.witness
            ))
        }
        let proofGenTime = (CFAbsoluteTimeGetCurrent() - proofGenT0) * 1000
        fputs(String(format: "  Proof generation (%d proofs): %.0f ms\n", maxN, proofGenTime), stderr)

        // ============================================================
        // MARK: - Correctness Tests
        // ============================================================

        fputs("\n--- Correctness Tests ---\n", stderr)
        var allCorrect = true

        // Test 1: Individual proof verification (baseline sanity)
        var indivOk = true
        for (i, item) in allItems.prefix(10).enumerated() {
            let g = pointFromAffine(srs[0])
            let vG = cPointScalarMul(g, item.value)
            let lhsP = pointAdd(item.commitment, pointNeg(vG))
            let sMz = frSub(srsSecret, item.point)
            let rhsP = cPointScalarMul(item.proof, sMz)
            let la = batchToAffine([lhsP])
            let ra = batchToAffine([rhsP])
            let ok = fpToInt(la[0].x) == fpToInt(ra[0].x) && fpToInt(la[0].y) == fpToInt(ra[0].y)
            if !ok {
                fputs("  [FAIL] Individual proof \(i) verification failed\n", stderr)
                indivOk = false
            }
        }
        fputs("  [\(indivOk ? "pass" : "FAIL")] All 10 individual proofs valid\n", stderr)
        allCorrect = allCorrect && indivOk

        // Test 2: Batch verify N=4, 16, 64 valid proofs (Fiat-Shamir)
        for n in [4, 16, 64] {
            guard n <= maxN else { continue }
            let items = Array(allItems.prefix(n))
            let batchOk = try batchVerifier.batchVerifyKZG(
                items: items, srs: srs, srsSecret: srsSecret)
            fputs("  [\(batchOk ? "pass" : "FAIL")] Batch verify \(n) valid proofs (Fiat-Shamir)\n", stderr)
            allCorrect = allCorrect && batchOk
        }

        // Test 3: Soundness -- tamper with one proof, batch should reject
        for n in [4, 16, 64] {
            guard n <= maxN else { continue }
            var tamperedItems = Array(allItems.prefix(n))
            let tamperIdx = n / 2
            let bogusValue = frAdd(tamperedItems[tamperIdx].value, frFromInt(1))
            tamperedItems[tamperIdx] = VerificationItem(
                commitment: tamperedItems[tamperIdx].commitment,
                point: tamperedItems[tamperIdx].point,
                value: bogusValue,
                proof: tamperedItems[tamperIdx].proof
            )
            let tamperedResult = try batchVerifier.batchVerifyKZG(
                items: tamperedItems, srs: srs, srsSecret: srsSecret)
            fputs("  [\(!tamperedResult ? "pass" : "FAIL")] Detect 1 invalid among \(n-1) valid (N=\(n))\n", stderr)
            allCorrect = allCorrect && !tamperedResult
        }

        // Test 4: Single proof batch verification
        let singleItems = [allItems[0]]
        let singleScalars = [frFromInt(1)]
        let singleOk = try batchVerifier.batchVerifyKZGWithScalars(
            items: singleItems, scalars: singleScalars, srs: srs, srsSecret: srsSecret)
        fputs("  [\(singleOk ? "pass" : "FAIL")] Single proof batch verification\n", stderr)
        allCorrect = allCorrect && singleOk

        // Test 5: GPU MSM batch verification (Fiat-Shamir)
        if maxN >= 64 {
            let msmItems = Array(allItems.prefix(64))
            let msmOk = try batchVerifier.batchVerifyKZGWithMSMFiatShamir(
                items: msmItems, srs: srs, srsSecret: srsSecret)
            fputs("  [\(msmOk ? "pass" : "FAIL")] GPU MSM batch verify 64 valid proofs\n", stderr)
            allCorrect = allCorrect && msmOk

            // Soundness with GPU MSM path
            var tamperedMSM = msmItems
            let bogus = frAdd(tamperedMSM[30].value, frFromInt(7))
            tamperedMSM[30] = VerificationItem(
                commitment: tamperedMSM[30].commitment,
                point: tamperedMSM[30].point,
                value: bogus,
                proof: tamperedMSM[30].proof
            )
            let tamperedMSMResult = try batchVerifier.batchVerifyKZGWithMSMFiatShamir(
                items: tamperedMSM, srs: srs, srsSecret: srsSecret)
            fputs("  [\(!tamperedMSMResult ? "pass" : "FAIL")] GPU MSM detect invalid (N=64)\n", stderr)
            allCorrect = allCorrect && !tamperedMSMResult
        }

        // Test 6: Adaptive path (auto-selects CPU or GPU based on batch size)
        for n in [4, 16, 64] {
            guard n <= maxN else { continue }
            let items = Array(allItems.prefix(n))
            let adaptiveOk = try batchVerifier.batchVerifyKZGAdaptive(
                items: items, srs: srs, srsSecret: srsSecret)
            let path = n >= BatchVerifier.gpuMSMThreshold ? "GPU" : "CPU"
            fputs("  [\(adaptiveOk ? "pass" : "FAIL")] Adaptive verify N=\(n) (\(path) path)\n", stderr)
            allCorrect = allCorrect && adaptiveOk
        }

        // Test 7: ProofAggregator API
        let aggregator = ProofAggregator(srs: srs, srsSecret: srsSecret)
        for item in allItems.prefix(16) {
            aggregator.addKZG(item)
        }
        let aggResult = try aggregator.verifyAll()
        fputs("  [\(aggResult ? "pass" : "FAIL")] ProofAggregator (16 KZG)\n", stderr)
        fputs("  Savings: \(aggregator.estimatedSavings)\n", stderr)
        allCorrect = allCorrect && aggResult

        // Test 8: ProofAggregator with tampered proof
        let aggTampered = ProofAggregator(srs: srs, srsSecret: srsSecret)
        for (i, item) in allItems.prefix(16).enumerated() {
            if i == 8 {
                let bad = VerificationItem(
                    commitment: item.commitment,
                    point: item.point,
                    value: frAdd(item.value, frFromInt(1)),
                    proof: item.proof)
                aggTampered.addKZG(bad)
            } else {
                aggTampered.addKZG(item)
            }
        }
        let aggTamperedResult = try aggTampered.verifyAll()
        fputs("  [\(!aggTamperedResult ? "pass" : "FAIL")] ProofAggregator detects invalid\n", stderr)
        allCorrect = allCorrect && !aggTamperedResult

        // Test 9: ProofAggregator detailed results
        let aggDetail = ProofAggregator(srs: srs, srsSecret: srsSecret)
        for item in allItems.prefix(8) {
            aggDetail.addKZG(item)
        }
        let detailed = try aggDetail.verifyAllDetailed()
        fputs("  [\(detailed.allValid ? "pass" : "FAIL")] Detailed: kzg=\(detailed.kzgValid) fri=\(detailed.friValid) ipa=\(detailed.ipaValid)\n", stderr)
        allCorrect = allCorrect && detailed.allValid

        fputs("\n  Correctness: \(allCorrect ? "ALL PASS" : "SOME FAILED")\n", stderr)

        // ============================================================
        // MARK: - Benchmark: Individual vs Batch vs GPU MSM
        // ============================================================

        fputs("\n--- Batch vs Individual Verification ---\n", stderr)
        fputs("  N        Individual    Batch CPU    Batch GPU  Speedup\n", stderr)

        for n in batchSizes {
            guard n <= maxN else { continue }
            let items = Array(allItems.prefix(n))

            // Individual verification: verify each proof separately
            let indivT0 = CFAbsoluteTimeGetCurrent()
            for item in items {
                let g = pointFromAffine(srs[0])
                let vG = cPointScalarMul(g, item.value)
                let lhsP = pointAdd(item.commitment, pointNeg(vG))
                let sMz = frSub(srsSecret, item.point)
                let rhsP = cPointScalarMul(item.proof, sMz)
                let _ = batchToAffine([lhsP])
                let _ = batchToAffine([rhsP])
            }
            let indivTime = (CFAbsoluteTimeGetCurrent() - indivT0) * 1000

            // CPU batch verification (Fiat-Shamir transcript)
            let cpuT0 = CFAbsoluteTimeGetCurrent()
            let _ = try batchVerifier.batchVerifyKZG(
                items: items, srs: srs, srsSecret: srsSecret)
            let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000

            // GPU MSM batch verification (Fiat-Shamir) -- only for N >= 16
            var gpuTime = -1.0
            if n >= BatchVerifier.gpuMSMThreshold {
                // Warmup
                let _ = try batchVerifier.batchVerifyKZGWithMSMFiatShamir(
                    items: items, srs: srs, srsSecret: srsSecret)
                let gpuT0 = CFAbsoluteTimeGetCurrent()
                let _ = try batchVerifier.batchVerifyKZGWithMSMFiatShamir(
                    items: items, srs: srs, srsSecret: srsSecret)
                gpuTime = (CFAbsoluteTimeGetCurrent() - gpuT0) * 1000
            }

            let bestBatch = gpuTime > 0 ? min(cpuTime, gpuTime) : cpuTime
            let speedup = indivTime / max(bestBatch, 0.001)
            let gpuStr = gpuTime > 0 ? String(format: "%9.1f ms", gpuTime) : "       n/a"
            fputs(String(format: "  %-8d %9.1f ms %9.1f ms ", n, indivTime, cpuTime) + gpuStr +
                  String(format: " %6.1fx\n", speedup), stderr)
        }

        // ============================================================
        // MARK: - GPU MSM Scaling
        // ============================================================

        fputs("\n--- GPU MSM Batch Verification Scaling ---\n", stderr)
        fputs("  N           Time     Per-proof\n", stderr)

        for n in batchSizes {
            guard n <= maxN, n >= BatchVerifier.gpuMSMThreshold else { continue }
            let items = Array(allItems.prefix(n))

            // Warmup
            let _ = try batchVerifier.batchVerifyKZGWithMSMFiatShamir(
                items: items, srs: srs, srsSecret: srsSecret)

            // Timed runs
            let runs = 3
            var times = [Double]()
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try batchVerifier.batchVerifyKZGWithMSMFiatShamir(
                    items: items, srs: srs, srsSecret: srsSecret)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[runs / 2]
            let perProof = median / Double(n)
            fputs(String(format: "  %-8d %9.1f ms %8.3f ms\n",
                         n, median, perProof), stderr)
        }

        // ============================================================
        // MARK: - ProofAggregator Throughput
        // ============================================================

        fputs("\n--- ProofAggregator Throughput ---\n", stderr)

        for n in batchSizes {
            guard n <= maxN else { continue }
            let agg = ProofAggregator(srs: srs, srsSecret: srsSecret)
            agg.addKZGBatch(Array(allItems.prefix(n)))

            // Warmup
            let _ = try agg.verifyAll()

            let t0 = CFAbsoluteTimeGetCurrent()
            let ok = try agg.verifyAll()
            let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            let proofsPerSec = Double(n) / (elapsed / 1000.0)
            fputs(String(format: "  N=%-4d: %.1f ms (%.0f proofs/sec) valid=", n, elapsed, proofsPerSec) +
                  (ok ? "yes" : "NO") + "\n", stderr)
        }

        // ============================================================
        // MARK: - Rollup Sequencer Simulation
        // ============================================================

        if !quick && maxN >= 256 {
            fputs("\n--- Rollup Sequencer Simulation (256 proofs/block) ---\n", stderr)

            let blockProofs = Array(allItems.prefix(256))

            // Simulate: individual verification (N separate checks)
            let seqT0 = CFAbsoluteTimeGetCurrent()
            for item in blockProofs {
                let g = pointFromAffine(srs[0])
                let vG = cPointScalarMul(g, item.value)
                let lhsP = pointAdd(item.commitment, pointNeg(vG))
                let sMz = frSub(srsSecret, item.point)
                let _ = cPointScalarMul(item.proof, sMz)
                let _ = batchToAffine([lhsP])
            }
            let seqTime = (CFAbsoluteTimeGetCurrent() - seqT0) * 1000

            // Simulate: batch verification via ProofAggregator
            let rollupAgg = ProofAggregator(srs: srs, srsSecret: srsSecret)
            rollupAgg.addKZGBatch(blockProofs)

            // Warmup
            let _ = try rollupAgg.verifyAll()

            let batchT0 = CFAbsoluteTimeGetCurrent()
            let batchOk = try rollupAgg.verifyAll()
            let batchTime = (CFAbsoluteTimeGetCurrent() - batchT0) * 1000

            let speedup = seqTime / max(batchTime, 0.001)
            fputs(String(format: "  Sequential (256 checks):    %.1f ms\n", seqTime), stderr)
            fputs(String(format: "  Batched (3 MSMs + 2 muls):  %.1f ms\n", batchTime), stderr)
            fputs(String(format: "  Speedup: %.1fx -- valid: ", speedup) + (batchOk ? "yes" : "NO") + "\n", stderr)
            fputs(String(format: "  Throughput: %.0f proofs/sec\n",
                         Double(256) / (batchTime / 1000.0)), stderr)
        }

        fputs("\nDone.\n", stderr)

    } catch {
        fputs("Error: \(error)\n", stderr)
    }
}
