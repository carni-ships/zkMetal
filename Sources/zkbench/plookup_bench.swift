// Plookup Lookup Argument Benchmark & Correctness Tests
import zkMetal
import Foundation

public func runPlookupBench() {
    fputs("\n=== Plookup Lookup Argument (GPU) ===\n", stderr)

    // --- Correctness Tests ---
    fputs("\n--- Correctness Tests ---\n", stderr)

    do {
        let engine = try GPUPlookupEngine()

        // Test 1: Simple lookup -- all elements from a small table
        let table1 = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let witness1 = [frFromInt(20), frFromInt(10), frFromInt(40), frFromInt(30)]
        let beta1 = frFromInt(12345)
        let gamma1 = frFromInt(67890)

        let proof1 = try engine.prove(witness: witness1, table: table1,
                                       beta: beta1, gamma: gamma1)
        let valid1 = engine.verify(proof: proof1, witness: witness1, table: table1)
        fputs("  Simple lookup (n=4, N=4): \(valid1 ? "PASS" : "FAIL")\n", stderr)

        // Test 2: Repeated lookups -- some table entries used multiple times
        let table2: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        let witness2: [Fr] = [1, 1, 3, 3, 5, 5, 7, 7].map { frFromInt($0) }

        let proof2 = try engine.prove(witness: witness2, table: table2)
        let valid2 = engine.verify(proof: proof2, witness: witness2, table: table2)
        fputs("  Repeated lookups (n=8, N=8): \(valid2 ? "PASS" : "FAIL")\n", stderr)

        // Test 3: All same value
        let table3: [Fr] = [frFromInt(42), frFromInt(99), frFromInt(7), frFromInt(13)]
        let witness3: [Fr] = [frFromInt(42), frFromInt(42), frFromInt(42), frFromInt(42)]

        let proof3 = try engine.prove(witness: witness3, table: table3)
        let valid3 = engine.verify(proof: proof3, witness: witness3, table: table3)
        fputs("  All-same witness (n=4, N=4): \(valid3 ? "PASS" : "FAIL")\n", stderr)

        // Test 4: Asymmetric sizes (n < N)
        let table4: [Fr] = (0..<16).map { frFromInt(UInt64($0 * 7 + 3)) }
        let witness4: [Fr] = (0..<4).map { table4[$0] }

        let proof4 = try engine.prove(witness: witness4, table: table4)
        let valid4 = engine.verify(proof: proof4, witness: witness4, table: table4)
        fputs("  Asymmetric (n=4, N=16): \(valid4 ? "PASS" : "FAIL")\n", stderr)

        // Test 5: Accumulator closes (Z[n] == 1)
        fputs("  Accumulator closes: \(frEqual(proof1.finalAccumulator, Fr.one) ? "PASS" : "FAIL")\n", stderr)
        fputs("  Sorted vector length: \(proof1.sortedVector.count == witness1.count + table1.count ? "PASS" : "FAIL")\n", stderr)

        // Test 6: Tampered proof should fail
        let tamperedProof = PlookupProof(
            sortedVector: proof1.sortedVector,
            accumulatorZ: proof1.accumulatorZ,
            beta: proof1.beta,
            gamma: proof1.gamma,
            finalAccumulator: frAdd(proof1.finalAccumulator, Fr.one)  // tamper
        )
        let rejected = !engine.verify(proof: tamperedProof, witness: witness1, table: table1)
        fputs("  Reject tampered accumulator: \(rejected ? "PASS" : "FAIL")\n", stderr)

    } catch {
        fputs("  ERROR: \(error)\n", stderr)
    }

    // --- Performance Tests ---
    if !skipCPU {
        fputs("\n--- Performance ---\n", stderr)
        do {
            let engine = try GPUPlookupEngine()
            engine.profile = true

            for logN in [8, 10, 12, 14] {
                let N = 1 << logN
                let n = N

                let table: [Fr] = (0..<N).map { frFromInt(UInt64($0 + 1)) }

                var rng: UInt64 = 0xDEAD_BEEF
                var witness = [Fr]()
                witness.reserveCapacity(n)
                for _ in 0..<n {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    let idx = Int(rng >> 32) % N
                    witness.append(table[idx])
                }

                let t0 = CFAbsoluteTimeGetCurrent()
                let proof = try engine.prove(witness: witness, table: table)
                let proveTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000

                let t1 = CFAbsoluteTimeGetCurrent()
                let valid = engine.verify(proof: proof, witness: witness, table: table)
                let verifyTime = (CFAbsoluteTimeGetCurrent() - t1) * 1000

                fputs("  Plookup 2^\(logN) (n=N=\(N)): prove \(String(format: "%.1f", proveTime))ms, verify \(String(format: "%.1f", verifyTime))ms -- \(valid ? "PASS" : "FAIL")\n", stderr)
            }
        } catch {
            fputs("  ERROR: \(error)\n", stderr)
        }
    }
}
