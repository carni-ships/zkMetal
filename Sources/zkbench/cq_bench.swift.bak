// cq (Cached Quotients) Lookup Argument Benchmark & Correctness Tests
import zkMetal
import Foundation

public func runCQLookupBench() {
    fputs("\n=== cq Cached Quotients Lookup ===\n", stderr)

    // Generate a test SRS
    let gx = fpFromInt(1)
    let gy = fpFromInt(2)
    let gen = PointAffine(x: gx, y: gy)
    let secret: [UInt32] = [0x1234, 0x5678, 0x9ABC, 0xDEF0, 0x1111, 0x2222, 0x3333, 0x0001]
    let srsSecret = frFromLimbs(secret)

    // --- Correctness Tests ---
    fputs("\n--- Correctness Tests ---\n", stderr)

    do {
        // Small SRS for correctness tests
        let srs = KZGEngine.generateTestSRS(secret: secret, size: 256, generator: gen)
        let engine = try CQEngine(srs: srs)

        // Test 1: Simple lookup — all 4 elements from a 4-element table
        let table1: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let tc1 = try engine.preprocessTable(table: table1)
        let lookups1: [Fr] = [frFromInt(20), frFromInt(10), frFromInt(40), frFromInt(30)]
        let proof1 = try engine.prove(lookups: lookups1, table: tc1)
        let valid1 = engine.verify(proof: proof1, table: tc1, numLookups: 4, srsSecret: srsSecret)
        fputs("  Simple lookup (N=4, |T|=4): \(valid1 ? "PASS" : "FAIL")\n", stderr)

        // Test 2: Repeated lookups
        let table2: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        let tc2 = try engine.preprocessTable(table: table2)
        let lookups2: [Fr] = [1, 1, 3, 3, 5, 5, 7, 7].map { frFromInt($0) }
        let proof2 = try engine.prove(lookups: lookups2, table: tc2)
        let valid2 = engine.verify(proof: proof2, table: tc2, numLookups: 8, srsSecret: srsSecret)
        fputs("  Repeated lookups (N=8, |T|=8): \(valid2 ? "PASS" : "FAIL")\n", stderr)

        // Test 3: Multiplicities check
        let mult = CQEngine.computeMultiplicities(table: table2, lookups: lookups2)
        let expected: [UInt64] = [2, 0, 2, 0, 2, 0, 2, 0]
        var multCorrect = true
        for i in 0..<8 {
            if !frEqual(mult[i], frFromInt(expected[i])) { multCorrect = false; break }
        }
        fputs("  Multiplicities correct: \(multCorrect ? "PASS" : "FAIL")\n", stderr)

        // Test 4: Asymmetric (N < |T|)
        let table3: [Fr] = (0..<16).map { frFromInt(UInt64($0 * 7 + 3)) }
        let tc3 = try engine.preprocessTable(table: table3)
        let lookups3: [Fr] = (0..<4).map { table3[$0] }
        let proof3 = try engine.prove(lookups: lookups3, table: tc3)
        let valid3 = engine.verify(proof: proof3, table: tc3, numLookups: 4, srsSecret: srsSecret)
        fputs("  Asymmetric (N=4, |T|=16): \(valid3 ? "PASS" : "FAIL")\n", stderr)

        // Test 5: N > |T| (many lookups into small table)
        let table4: [Fr] = [frFromInt(100), frFromInt(200), frFromInt(300), frFromInt(400)]
        let tc4 = try engine.preprocessTable(table: table4)
        var lookups4 = [Fr]()
        for i in 0..<16 {
            lookups4.append(table4[i % 4])
        }
        let proof4 = try engine.prove(lookups: lookups4, table: tc4)
        let valid4 = engine.verify(proof: proof4, table: tc4, numLookups: 16, srsSecret: srsSecret)
        fputs("  N > |T| (N=16, |T|=4): \(valid4 ? "PASS" : "FAIL")\n", stderr)

        // Test 6: Wrong multiplicity count should fail verification
        let wrongProof = CQProof(
            hCommitment: proof1.hCommitment,
            multiplicities: proof1.multiplicities,
            multiplicitySum: frAdd(proof1.multiplicitySum, Fr.one),
            challengeZ: proof1.challengeZ,
            hOpening: proof1.hOpening,
            tOpening: proof1.tOpening,
            hEvalAtZ: proof1.hEvalAtZ,
            tEvalAtZ: proof1.tEvalAtZ
        )
        let rejected = !engine.verify(proof: wrongProof, table: tc1, numLookups: 4, srsSecret: srsSecret)
        fputs("  Reject tampered sum: \(rejected ? "PASS" : "FAIL")\n", stderr)

        // Test 7: Larger table (|T|=64, N=32)
        let table5: [Fr] = (0..<64).map { frFromInt(UInt64($0 * 11 + 5)) }
        let tc5 = try engine.preprocessTable(table: table5)
        var rng: UInt64 = 0xCAFE_BABE
        var lookups5 = [Fr]()
        for _ in 0..<32 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            let idx = Int(rng >> 32) % 64
            lookups5.append(table5[idx])
        }
        let proof5 = try engine.prove(lookups: lookups5, table: tc5)
        let valid5 = engine.verify(proof: proof5, table: tc5, numLookups: 32, srsSecret: srsSecret)
        fputs("  Larger (N=32, |T|=64): \(valid5 ? "PASS" : "FAIL")\n", stderr)

    } catch {
        fputs("  ERROR: \(error)\n", stderr)
    }

    // --- Performance Tests ---
    fputs("\n--- Performance: cq vs LogUp ---\n", stderr)
    fputs("  (cq prover time should be independent of |T|)\n\n", stderr)

    do {
        let lookupLogSizes = [10, 14]
        let tableLogSizes = [8, 12, 16]

        let maxTableSize = 1 << tableLogSizes.max()!
        let maxSRSSize = maxTableSize + 1
        fputs("  Generating SRS of size \(maxSRSSize)...\n", stderr)
        let srs = KZGEngine.generateTestSRS(secret: secret, size: maxSRSSize, generator: gen)

        let cqEngine = try CQEngine(srs: srs)
        let logupEngine = try LookupEngine()

        for lookupLog in lookupLogSizes {
            let N = 1 << lookupLog
            fputs("  N = 2^\(lookupLog) = \(N) lookups:\n", stderr)

            for tableLog in tableLogSizes {
                let T = 1 << tableLog
                if T < N { continue }

                // Build table
                let table: [Fr] = (0..<T).map { frFromInt(UInt64($0 + 1)) }

                // Generate random lookups (all valid)
                var rng: UInt64 = UInt64(lookupLog) &* 0xDEAD &+ UInt64(tableLog) &* 0xBEEF
                var lookups = [Fr]()
                lookups.reserveCapacity(N)
                for _ in 0..<N {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    let idx = Int(rng >> 32) % T
                    lookups.append(table[idx])
                }

                // --- cq ---
                let prepT0 = CFAbsoluteTimeGetCurrent()
                let tc = try cqEngine.preprocessTable(table: table)
                let prepTime = (CFAbsoluteTimeGetCurrent() - prepT0) * 1000

                let proveT0 = CFAbsoluteTimeGetCurrent()
                let cqProof = try cqEngine.prove(lookups: lookups, table: tc)
                let cqProveTime = (CFAbsoluteTimeGetCurrent() - proveT0) * 1000

                let verifyT0 = CFAbsoluteTimeGetCurrent()
                let cqValid = cqEngine.verify(
                    proof: cqProof, table: tc, numLookups: N, srsSecret: srsSecret)
                let cqVerifyTime = (CFAbsoluteTimeGetCurrent() - verifyT0) * 1000

                // --- LogUp ---
                let beta = frFromInt(UInt64(lookupLog + tableLog) &* 31337)

                let logupT0 = CFAbsoluteTimeGetCurrent()
                let logupProof = try logupEngine.prove(
                    table: table, lookups: lookups, beta: beta)
                let logupProveTime = (CFAbsoluteTimeGetCurrent() - logupT0) * 1000

                let logupV0 = CFAbsoluteTimeGetCurrent()
                let logupValid = try logupEngine.verify(
                    proof: logupProof, table: table, lookups: lookups)
                let logupVerifyTime = (CFAbsoluteTimeGetCurrent() - logupV0) * 1000

                fputs(String(format: "    |T|=2^%-2d: cq preprocess %6.1fms, prove %6.1fms, verify %6.1fms %s\n",
                             tableLog, prepTime, cqProveTime, cqVerifyTime,
                             cqValid ? "OK" : "FAIL"), stderr)
                fputs(String(format: "    %*s  LogUp prove %6.1fms, verify %6.1fms %s\n",
                             10, "", logupProveTime, logupVerifyTime,
                             logupValid ? "OK" : "FAIL"), stderr)

                if logupProveTime > 0 {
                    let speedup = logupProveTime / cqProveTime
                    fputs(String(format: "    %*s  cq/LogUp prove ratio: %.2fx\n",
                                 10, "", speedup), stderr)
                }
            }
            fputs("\n", stderr)
        }
    } catch {
        fputs("  ERROR: \(error)\n", stderr)
    }
}
