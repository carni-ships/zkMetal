// Lasso Structured Lookup Benchmark & Correctness Tests
import zkMetal
import Foundation

public func runLassoBench() {
    fputs("\n=== Lasso Structured Lookup ===\n", stderr)

    // --- Correctness Tests ---
    fputs("\n--- Correctness Tests ---\n", stderr)

    do {
        let engine = try LassoEngine()

        // Test 1: Range check [0, 256) with 1 chunk (trivial decomposition)
        let table1 = LassoTable.rangeCheck(bits: 8, chunks: 1)
        let lookups1: [Fr] = [frFromInt(0), frFromInt(127), frFromInt(255), frFromInt(42)]
        let proof1 = try engine.prove(lookups: lookups1, table: table1)
        let valid1 = try engine.verify(proof: proof1, lookups: lookups1, table: table1)
        fputs("  Range [0,256) 1-chunk, m=4: \(valid1 ? "PASS" : "FAIL")\n", stderr)

        // Test 2: Range check [0, 2^16) with 2 byte chunks
        let table2 = LassoTable.rangeCheck(bits: 16, chunks: 2)
        let lookups2: [Fr] = [frFromInt(0), frFromInt(255), frFromInt(256), frFromInt(65535)]
        let proof2 = try engine.prove(lookups: lookups2, table: table2)
        let valid2 = try engine.verify(proof: proof2, lookups: lookups2, table: table2)
        fputs("  Range [0,2^16) 2-chunk, m=4: \(valid2 ? "PASS" : "FAIL")\n", stderr)

        // Test 3: Range check [0, 2^32) with 4 byte chunks (the canonical Lasso use case)
        let table3 = LassoTable.rangeCheck(bits: 32, chunks: 4)
        let lookups3: [Fr] = [frFromInt(0), frFromInt(1000), frFromInt(65535), frFromInt(100000)]
        let proof3 = try engine.prove(lookups: lookups3, table: table3)
        let valid3 = try engine.verify(proof: proof3, lookups: lookups3, table: table3)
        fputs("  Range [0,2^32) 4-chunk, m=4: \(valid3 ? "PASS" : "FAIL")\n", stderr)

        // Test 4: Larger lookup set with repeated values
        let table4 = LassoTable.rangeCheck(bits: 16, chunks: 2)
        var lookups4 = [Fr]()
        for i in 0..<16 {
            lookups4.append(frFromInt(UInt64(i * 1000)))
        }
        let proof4 = try engine.prove(lookups: lookups4, table: table4)
        let valid4 = try engine.verify(proof: proof4, lookups: lookups4, table: table4)
        fputs("  Range [0,2^16) 2-chunk, m=16: \(valid4 ? "PASS" : "FAIL")\n", stderr)

        // Test 5: Verify decomposition correctness — tampered indices should fail
        let tamperedProof = LassoProof(
            numChunks: proof2.numChunks,
            subtableProofs: proof2.subtableProofs,
            indices: proof2.indices.map { $0.map { ($0 + 1) % 256 } }  // shift all indices
        )
        let rejected = try !engine.verify(proof: tamperedProof, lookups: lookups2, table: table2)
        fputs("  Reject tampered indices: \(rejected ? "PASS" : "FAIL")\n", stderr)

        // Test 6: Verify that wrong lookups fail
        let wrongLookups: [Fr] = [frFromInt(0), frFromInt(1), frFromInt(2), frFromInt(3)]
        let rejected2 = try !engine.verify(proof: proof2, lookups: wrongLookups, table: table2)
        fputs("  Reject wrong lookups: \(rejected2 ? "PASS" : "FAIL")\n", stderr)

        // Test 7: XOR/AND tables require operand-aware decomposition (the lookup
        // value is a packed index (a,b), not just the result). The current LassoTable
        // API with value-based decomposition is designed for range-check-style tables.
        // XOR/AND are provided as table constructors but need the caller to provide
        // packed indices as lookup values rather than operation results.
        fputs("  XOR/AND tables: designed for index-based lookups (see docs)\n", stderr)

        // (XOR/AND table tests skipped — see note above about operand-aware decomposition)

    } catch {
        fputs("  ERROR: \(error)\n", stderr)
    }

    // --- Profile: identify phase costs at 2^18 ---
    fputs("\n--- Profile (2^18) ---\n", stderr)
    do {
        let profEngine = try LassoEngine()
        profEngine.profileLasso = true
        let profM = 1 << 18
        var rngP: UInt64 = 0xDEAD_BEEF_1234
        var profValues = [UInt64]()
        profValues.reserveCapacity(profM)
        for _ in 0..<profM {
            rngP = rngP &* 6364136223846793005 &+ 1442695040888963407
            profValues.append(rngP >> 32)
        }
        let profLookups: [Fr] = profValues.map { frFromInt($0) }
        let profTable = LassoTable.rangeCheck(bits: 32, chunks: 4)
        // Warmup
        let _ = try profEngine.prove(lookups: profLookups, table: profTable)
        fputs("  --- profiled run ---\n", stderr)
        let _ = try profEngine.prove(lookups: profLookups, table: profTable)
        profEngine.profileLasso = false
    } catch {
        fputs("  Profile ERROR: \(error)\n", stderr)
    }

    // --- Performance: Lasso vs LogUp ---
    fputs("\n--- Performance: Lasso vs LogUp ---\n", stderr)
    do {
        let lassoEngine = try LassoEngine()
        let logupEngine = try LookupEngine()

        for logM in [10, 14, 18] {
            let m = 1 << logM

            // Generate random values in [0, 2^32) for range check
            var rng: UInt64 = 0xDEAD_BEEF_1234
            var values = [UInt64]()
            values.reserveCapacity(m)
            for _ in 0..<m {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                values.append(rng >> 32)  // 32-bit range
            }
            let lookups: [Fr] = values.map { frFromInt($0) }

            // Lasso: range check [0, 2^32) decomposed into 4 byte subtables
            let lassoTable = LassoTable.rangeCheck(bits: 32, chunks: 4)

            // Warmup
            let _ = try lassoEngine.prove(lookups: lookups, table: lassoTable)

            // Timed Lasso prove
            let runs = 3
            var lassoTimes = [Double]()
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                let proof = try lassoEngine.prove(lookups: lookups, table: lassoTable)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                lassoTimes.append(elapsed)

                // Verify on first run
                if lassoTimes.count == 1 {
                    let vt0 = CFAbsoluteTimeGetCurrent()
                    let valid = try lassoEngine.verify(proof: proof, lookups: lookups, table: lassoTable)
                    let verifyTime = (CFAbsoluteTimeGetCurrent() - vt0) * 1000
                    if !valid {
                        fputs("  WARNING: Lasso verification FAILED for 2^\(logM)\n", stderr)
                    }
                    fputs(String(format: "  Lasso 2^%-2d (m=%d): prove %.1fms, verify %.1fms\n",
                                logM, m, elapsed, verifyTime), stderr)
                }
            }
            lassoTimes.sort()
            let lassoMedian = lassoTimes[runs / 2]

            // LogUp comparison: need full table [0, 2^32) — only feasible for small sizes
            if logM <= 14 {
                // For fair comparison with LogUp, use a 256-entry table (same as one Lasso subtable)
                // and range check [0, 256) instead
                let smallValues: [Fr] = values.prefix(m).map { frFromInt($0 % 256) }
                let table256: [Fr] = (0..<256).map { frFromInt(UInt64($0)) }
                let beta = frFromInt(12345)

                // Warmup
                let _ = try logupEngine.prove(table: table256, lookups: smallValues, beta: beta)

                var logupTimes = [Double]()
                for _ in 0..<runs {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let _ = try logupEngine.prove(table: table256, lookups: smallValues, beta: beta)
                    let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                    logupTimes.append(elapsed)
                }
                logupTimes.sort()
                let logupMedian = logupTimes[runs / 2]

                fputs(String(format: "  LogUp 2^%-2d (m=%d, T=256): prove %.1fms\n",
                            logM, m, logupMedian), stderr)
                fputs(String(format: "  Lasso median: %.1fms (range [0,2^32) via 4 subtables of 256)\n",
                            lassoMedian), stderr)
            } else {
                fputs(String(format: "  Lasso 2^%-2d median: %.1fms (range [0,2^32) via 4 subtables)\n",
                            logM, lassoMedian), stderr)
                fputs("  LogUp: infeasible (would need 2^32 table entries)\n", stderr)
            }
            fputs("\n", stderr)
        }
    } catch {
        fputs("  ERROR: \(error)\n", stderr)
    }
}
