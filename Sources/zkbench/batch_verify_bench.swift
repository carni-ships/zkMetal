// Batch Verification Benchmark
// Compares: individual verification (N scalar checks) vs batch verification (random LC)
// Tests correctness: batch detects a single invalid proof among N-1 valid ones

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
        let batchSizes = quick ? [10, 50] : [10, 50, 100, 500]
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

        // MARK: - Correctness test: individual verification

        fputs("\n--- Correctness Tests ---\n", stderr)

        // First verify each proof individually to ensure they are valid
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
        fputs("  [\(indivOk ? "pass" : "FAIL")] All 10 individual proofs valid: \(indivOk)\n", stderr)

        // Test batch verification with simple scalars
        let testItems = Array(allItems.prefix(10))
        var testScalars = [Fr]()
        for i in 0..<10 {
            testScalars.append(frFromInt(UInt64(i + 1)))
        }

        let batchOk = try batchVerifier.batchVerifyKZGWithScalars(
            items: testItems, scalars: testScalars, srs: srs, srsSecret: srsSecret)
        fputs("  [\(batchOk ? "pass" : "FAIL")] Batch verify 10 valid proofs: \(batchOk)\n", stderr)

        // Test 2: Tamper with one proof -- should fail batch
        var tamperedItems = testItems
        let bogusValue = frAdd(tamperedItems[5].value, frFromInt(1))
        tamperedItems[5] = VerificationItem(
            commitment: tamperedItems[5].commitment,
            point: tamperedItems[5].point,
            value: bogusValue,
            proof: tamperedItems[5].proof
        )
        let tamperedResult = try batchVerifier.batchVerifyKZGWithScalars(
            items: tamperedItems, scalars: testScalars, srs: srs, srsSecret: srsSecret)
        fputs("  [\(tamperedResult ? "FAIL" : "pass")] Detect 1 invalid among 9 valid: \(!tamperedResult)\n", stderr)

        // Test 3: Single proof verification
        let singleItems = [allItems[0]]
        let singleScalars = [frFromInt(1)]
        let singleOk = try batchVerifier.batchVerifyKZGWithScalars(
            items: singleItems, scalars: singleScalars, srs: srs, srsSecret: srsSecret)
        fputs("  [\(singleOk ? "pass" : "FAIL")] Single proof verification: \(singleOk)\n", stderr)

        // Test 4: ProofAggregator API
        let aggregator = ProofAggregator(srs: srs, srsSecret: srsSecret)
        for item in testItems {
            aggregator.addKZG(item)
        }
        let aggResult = try aggregator.verifyAll()
        fputs("  [\(aggResult ? "pass" : "FAIL")] ProofAggregator (10 KZG): \(aggResult)\n", stderr)
        fputs("  Savings: \(aggregator.estimatedSavings)\n", stderr)

        // MARK: - Benchmark: individual vs batch

        fputs("\n--- Batch vs Individual Verification ---\n", stderr)
        fputs(String(format: "  %-8s %12s %12s %8s\n", "N", "Individual", "Batch", "Speedup"), stderr)

        for n in batchSizes {
            let items = Array(allItems.prefix(n))

            // Individual verification: verify each proof separately using SRS secret
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

            // Batch verification
            var batchScalars = [Fr]()
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                batchScalars.append(frFromInt(rng >> 32))
            }
            let batchT0 = CFAbsoluteTimeGetCurrent()
            let _ = try batchVerifier.batchVerifyKZGWithScalars(
                items: items, scalars: batchScalars, srs: srs, srsSecret: srsSecret)
            let batchTime = (CFAbsoluteTimeGetCurrent() - batchT0) * 1000

            let speedup = indivTime / max(batchTime, 0.001)
            fputs(String(format: "  %-8d %9.1f ms %9.1f ms %6.1fx\n",
                         n, indivTime, batchTime, speedup), stderr)
        }

        // MARK: - GPU MSM batch (for larger sizes)

        if !quick && maxN >= 100 {
            fputs("\n--- GPU MSM Batch Verification ---\n", stderr)
            for n in [100, 500] {
                guard n <= maxN else { continue }
                let items = Array(allItems.prefix(n))
                var msmScalars = [Fr]()
                for _ in 0..<n {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    msmScalars.append(frFromInt(rng >> 32))
                }

                // Warmup
                let _ = try batchVerifier.batchVerifyKZGWithMSM(
                    items: items, scalars: msmScalars, srs: srs, srsSecret: srsSecret)

                let msmT0 = CFAbsoluteTimeGetCurrent()
                let msmOk = try batchVerifier.batchVerifyKZGWithMSM(
                    items: items, scalars: msmScalars, srs: srs, srsSecret: srsSecret)
                let msmTime = (CFAbsoluteTimeGetCurrent() - msmT0) * 1000
                fputs(String(format: "  GPU MSM batch verify N=%-4d: %.1f ms (valid: %@)\n",
                             n, msmTime, msmOk ? "yes" : "NO"), stderr)
            }
        }

        fputs("\nDone.\n", stderr)

    } catch {
        fputs("Error: \(error)\n", stderr)
    }
}
