// LogUp Lookup Argument Benchmark & Correctness Tests
import zkMetal
import Foundation

public func runLookupBench() {
    fputs("\n=== LogUp Lookup Argument ===\n", stderr)

    // --- Correctness Tests ---
    fputs("\n--- Correctness Tests ---\n", stderr)

    do {
        let engine = try LookupEngine()

        // Test 1: Simple lookup — all elements from a small table
        let table = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]  // N=4
        let lookups = [frFromInt(20), frFromInt(10), frFromInt(40), frFromInt(30)]  // m=4, all in table
        let beta = frFromInt(12345)

        let proof = try engine.prove(table: table, lookups: lookups, beta: beta)
        let valid = try engine.verify(proof: proof, table: table, lookups: lookups)
        fputs("  Simple lookup (m=4, N=4): \(valid ? "PASS" : "FAIL")\n", stderr)

        // Test 2: Repeated lookups — some table entries used multiple times
        let table2: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }  // T = [1..8]
        let lookups2: [Fr] = [1, 1, 3, 3, 5, 5, 7, 7].map { frFromInt($0) }  // m=8, repeats
        let beta2 = frFromInt(99999)

        let proof2 = try engine.prove(table: table2, lookups: lookups2, beta: beta2)
        let valid2 = try engine.verify(proof: proof2, table: table2, lookups: lookups2)
        fputs("  Repeated lookups (m=8, N=8): \(valid2 ? "PASS" : "FAIL")\n", stderr)

        // Test 3: Check multiplicities
        let mult = LookupEngine.computeMultiplicities(table: table2, lookups: lookups2)
        let expected: [UInt64] = [2, 0, 2, 0, 2, 0, 2, 0]  // 1 appears 2x, 2 appears 0x, etc.
        var multCorrect = true
        for i in 0..<8 {
            if !frEqual(mult[i], frFromInt(expected[i])) {
                multCorrect = false
                break
            }
        }
        fputs("  Multiplicities correct: \(multCorrect ? "PASS" : "FAIL")\n", stderr)

        // Test 4: Batch inverse correctness
        let poly = try PolyEngine()
        let testVals = [frFromInt(3), frFromInt(7), frFromInt(11), frFromInt(13)]
        let inverses = try poly.batchInverse(testVals)
        var allInvCorrect = true
        for i in 0..<4 {
            let product = frMul(testVals[i], inverses[i])
            if !frEqual(product, Fr.one) {
                allInvCorrect = false
                fputs("  Batch inverse at \(i): product != 1\n", stderr)
            }
        }
        fputs("  Batch inverse: \(allInvCorrect ? "PASS" : "FAIL")\n", stderr)

        // Test 5: Larger lookup (m=16, N=16)
        let table3: [Fr] = (0..<16).map { frFromInt(UInt64($0 * 7 + 3)) }
        var lookups3 = [Fr]()
        for i in 0..<16 {
            lookups3.append(table3[i % 16])  // all valid
        }
        let beta3 = frFromInt(777)
        let proof3 = try engine.prove(table: table3, lookups: lookups3, beta: beta3)
        let valid3 = try engine.verify(proof: proof3, table: table3, lookups: lookups3)
        fputs("  Larger lookup (m=16, N=16): \(valid3 ? "PASS" : "FAIL")\n", stderr)

        // Test 6: Lookup with m != N (m=8 lookups into N=16 table)
        let lookups4: [Fr] = (0..<8).map { table3[$0] }
        let beta4 = frFromInt(54321)
        let proof4 = try engine.prove(table: table3, lookups: lookups4, beta: beta4)
        let valid4 = try engine.verify(proof: proof4, table: table3, lookups: lookups4)
        fputs("  Asymmetric lookup (m=8, N=16): \(valid4 ? "PASS" : "FAIL")\n", stderr)

        // Test 7: Wrong beta should fail verification
        var wrongProof = proof
        // Tamper: change the claimed sum
        let tamperedSum = frAdd(proof.claimedSum, Fr.one)
        wrongProof = LookupProof(
            multiplicities: proof.multiplicities,
            beta: proof.beta,
            lookupSumcheckRounds: proof.lookupSumcheckRounds,
            tableSumcheckRounds: proof.tableSumcheckRounds,
            claimedSum: tamperedSum,
            lookupFinalEval: proof.lookupFinalEval,
            tableFinalEval: proof.tableFinalEval
        )
        let rejected = try !engine.verify(proof: wrongProof, table: table, lookups: lookups)
        fputs("  Reject tampered sum: \(rejected ? "PASS" : "FAIL")\n", stderr)

    } catch {
        fputs("  ERROR: \(error)\n", stderr)
    }

    // --- Range Proof Tests ---
    fputs("\n--- Range Proof ---\n", stderr)
    do {
        let rpe = try RangeProofEngine()

        // Test: small range, direct lookup
        let vals1: [UInt64] = [0, 3, 7, 15]
        let rp1 = try rpe.prove(values: vals1, range: 16)
        let rv1 = try rpe.verify(proof: rp1, values: vals1)
        fputs("  Direct range [0,16), n=4: \(rv1 ? "PASS" : "FAIL") (decomposed=\(rp1.decomposed))\n", stderr)

        // Test: 8-bit range
        let vals2: [UInt64] = [0, 127, 255, 200, 50, 100, 150, 1]
        let rp2 = try rpe.prove(values: vals2, range: 256)
        let rv2 = try rpe.verify(proof: rp2, values: vals2)
        fputs("  Direct range [0,256), n=8: \(rv2 ? "PASS" : "FAIL")\n", stderr)

        // Test: large range requiring decomposition (32-bit)
        let vals3: [UInt64] = [0, 1000, 65535, 100000]
        let rp3 = try rpe.prove(values: vals3, range: 1 << 32)
        let rv3 = try rpe.verify(proof: rp3, values: vals3)
        fputs("  Decomposed range [0,2^32), n=4, limbs=\(rp3.numLimbs): \(rv3 ? "PASS" : "FAIL")\n", stderr)

        // Test: 16-bit range (boundary — still direct)
        let vals4: [UInt64] = Array(0..<16).map { $0 * 4096 + 1 }
        let rp4 = try rpe.prove(values: vals4, range: 1 << 16)
        let rv4 = try rpe.verify(proof: rp4, values: vals4)
        fputs("  Direct range [0,2^16), n=16: \(rv4 ? "PASS" : "FAIL")\n", stderr)

    } catch {
        fputs("  ERROR: \(error)\n", stderr)
    }

    // --- Performance Tests ---
    if !skipCPU {
        fputs("\n--- Performance ---\n", stderr)
        do {
            let engine = try LookupEngine()
            engine.profileLogUp = true

            for logN in [8, 10, 12, 14] {
                let N = 1 << logN
                let m = N  // same size for simplicity

                // Build table with distinct values
                let table: [Fr] = (0..<N).map { frFromInt(UInt64($0 + 1)) }

                // Random lookups (all valid, with repetition)
                var rng: UInt64 = 0xDEAD_BEEF
                var lookups = [Fr]()
                lookups.reserveCapacity(m)
                for _ in 0..<m {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    let idx = Int(rng >> 32) % N
                    lookups.append(table[idx])
                }

                let beta = frFromInt(UInt64(logN) &* 31337)

                let t0 = CFAbsoluteTimeGetCurrent()
                let proof = try engine.prove(table: table, lookups: lookups, beta: beta)
                let proveTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000

                let t1 = CFAbsoluteTimeGetCurrent()
                let valid = try engine.verify(proof: proof, table: table, lookups: lookups)
                let verifyTime = (CFAbsoluteTimeGetCurrent() - t1) * 1000

                fputs("  LogUp 2^\(logN) (m=N=\(N)): prove \(String(format: "%.1f", proveTime))ms, verify \(String(format: "%.1f", verifyTime))ms — \(valid ? "PASS" : "FAIL")\n", stderr)
            }
        } catch {
            fputs("  ERROR: \(error)\n", stderr)
        }
    }
}
