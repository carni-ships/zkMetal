// Batch Verification Benchmark
// Compares: individual verification (N scalar checks) vs batch verification (random LC)
// Tests correctness: batch detects a single invalid proof among N-1 valid ones

import Foundation
import zkMetal

public func runBatchVerifyBench() {
    fputs("=== Batch KZG Verification Benchmark ===\n", stderr)

    do {
        // Setup: generate SRS
        fputs("Generating SRS...\n", stderr)
        let generator = PointAffine(x: fpFromInt(1), y: fpFromInt(2))
        let secret: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]
        let srsSize = 1024
        let srs = KZGEngine.generateTestSRS(secret: secret, size: srsSize, generator: generator)
        fputs("SRS generated\n", stderr)
        let srsSecret = frFromLimbs(secret)
        let kzg = try KZGEngine(srs: srs)
        fputs("KZG engine initialized\n", stderr)
        let batchVerifier = try BatchVerifier(msmEngine: kzg.msmEngine)
        fputs("Batch verifier initialized\n", stderr)

        // Generate random polynomials and their KZG proofs
        let quick = CommandLine.arguments.contains("--quick")
        let batchSizes = quick ? [10, 50] : [10, 50, 100, 500]
        let maxN = batchSizes.last!

        fputs("Generating \(maxN) KZG proofs...\n", stderr)
        var rng: UInt64 = 0xCAFE_BABE_DEAD_BEEF
        func nextRng() -> UInt64 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return rng >> 32
        }

        var allItems = [VerificationItem]()
        allItems.reserveCapacity(maxN)

        let proofGenT0 = CFAbsoluteTimeGetCurrent()
        for idx in 0..<maxN {
            if idx % 10 == 0 { fputs("  proof \(idx)/\(maxN)\n", stderr) }
            // Random polynomial of degree 15
            let degree = 16
            var coeffs = [Fr]()
            coeffs.reserveCapacity(degree)
            for _ in 0..<degree {
                coeffs.append(frFromInt(nextRng()))
            }

            // Random evaluation point
            let z = frFromInt(nextRng())

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
        print(String(format: "  Proof generation (%d proofs): %.0f ms", maxN, proofGenTime))

        // MARK: - Correctness test

        print("\n--- Correctness Tests ---")

        // Test 1: All valid proofs should verify
        let testItems = Array(allItems.prefix(10))
        let scalars = (0..<10).map { _ in frFromInt(nextRng()) }

        let allValid = try batchVerifier.batchVerifyKZGWithScalars(
            items: testItems, scalars: scalars, srs: srs, srsSecret: srsSecret)
        print("  [\(allValid ? "pass" : "FAIL")] Batch verify 10 valid proofs: \(allValid)")

        // Test 2: Tamper with one proof -- should fail batch
        var tamperedItems = testItems
        // Corrupt the value of proof #5
        let originalValue = tamperedItems[5].value
        let bogusValue = frAdd(originalValue, frFromInt(1))
        tamperedItems[5] = VerificationItem(
            commitment: tamperedItems[5].commitment,
            point: tamperedItems[5].point,
            value: bogusValue,
            proof: tamperedItems[5].proof
        )
        let tamperedResult = try batchVerifier.batchVerifyKZGWithScalars(
            items: tamperedItems, scalars: scalars, srs: srs, srsSecret: srsSecret)
        print("  [\(tamperedResult ? "FAIL" : "pass")] Detect 1 invalid among 9 valid: \(!tamperedResult)")

        // Test 3: Single proof verification
        let singleItem = [allItems[0]]
        let singleScalar = [frFromInt(1)]
        let singleValid = try batchVerifier.batchVerifyKZGWithScalars(
            items: singleItem, scalars: singleScalar, srs: srs, srsSecret: srsSecret)
        print("  [\(singleValid ? "pass" : "FAIL")] Single proof verification: \(singleValid)")

        // Test 4: ProofAggregator API
        let aggregator = ProofAggregator(srs: srs, srsSecret: srsSecret)
        for item in testItems {
            aggregator.addKZG(item)
        }
        let aggResult = try aggregator.verifyAll()
        print("  [\(aggResult ? "pass" : "FAIL")] ProofAggregator (10 KZG): \(aggResult)")
        print("  Savings: \(aggregator.estimatedSavings)")

        // MARK: - Benchmark: individual vs batch

        print("\n--- Batch vs Individual Verification ---")
        print(String(format: "  %-8s %12s %12s %8s", "N", "Individual", "Batch", "Speedup"))

        for n in batchSizes {
            let items = Array(allItems.prefix(n))

            // Individual verification: verify each proof separately using SRS secret
            let indivT0 = CFAbsoluteTimeGetCurrent()
            var indivAllOk = true
            for item in items {
                // Per-proof check: C - v*G == (s-z)*pi
                let g = pointFromAffine(srs[0])
                let vG = cPointScalarMul(g, item.value)
                let lhs = pointAdd(item.commitment, pointNeg(vG))
                let sMz = frSub(srsSecret, item.point)
                let rhs = cPointScalarMul(item.proof, sMz)
                let la = batchToAffine([lhs])
                let ra = batchToAffine([rhs])
                if fpToInt(la[0].x) != fpToInt(ra[0].x) || fpToInt(la[0].y) != fpToInt(ra[0].y) {
                    indivAllOk = false
                }
            }
            let indivTime = (CFAbsoluteTimeGetCurrent() - indivT0) * 1000
            _ = indivAllOk

            // Batch verification
            let batchT0 = CFAbsoluteTimeGetCurrent()
            let batchScalars = (0..<n).map { _ in frFromInt(nextRng()) }
            let batchOk = try batchVerifier.batchVerifyKZGWithScalars(
                items: items, scalars: batchScalars, srs: srs, srsSecret: srsSecret)
            let batchTime = (CFAbsoluteTimeGetCurrent() - batchT0) * 1000
            _ = batchOk

            let speedup = indivTime / max(batchTime, 0.001)
            print(String(format: "  %-8d %9.1f ms %9.1f ms %6.1fx",
                         n, indivTime, batchTime, speedup))
        }

        // MARK: - GPU MSM batch (for larger sizes)

        if !quick && maxN >= 100 {
            print("\n--- GPU MSM Batch Verification ---")
            for n in [100, 500] {
                guard n <= maxN else { continue }
                let items = Array(allItems.prefix(n))
                let msmScalars = (0..<n).map { _ in frFromInt(nextRng()) }

                // Warmup
                let _ = try batchVerifier.batchVerifyKZGWithMSM(
                    items: items, scalars: msmScalars, srs: srs, srsSecret: srsSecret)

                let msmT0 = CFAbsoluteTimeGetCurrent()
                let msmOk = try batchVerifier.batchVerifyKZGWithMSM(
                    items: items, scalars: msmScalars, srs: srs, srsSecret: srsSecret)
                let msmTime = (CFAbsoluteTimeGetCurrent() - msmT0) * 1000
                print(String(format: "  GPU MSM batch verify N=%-4d: %.1f ms (valid: %@)",
                             n, msmTime, msmOk ? "yes" : "NO"))
            }
        }

        print("\nDone.")

    } catch {
        print("Error: \(error)")
    }
}
