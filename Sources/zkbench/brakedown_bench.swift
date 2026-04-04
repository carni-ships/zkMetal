// Brakedown Polynomial Commitment Benchmark and Correctness Tests
import zkMetal
import Foundation

public func runBrakedownBench() {
    print("=== Brakedown Polynomial Commitment Benchmark ===")
    fflush(stdout)
    print("NTT-free PCS using random linear codes + Merkle commitments")
    fflush(stdout)

    do {
        print("Creating engine...")
        fflush(stdout)
        let engine = try BrakedownEngine(rateInverse: 4, numQueries: 30)
        print("Engine created.")
        fflush(stdout)

        // --- Correctness Tests ---
        print("\n--- Correctness verification ---")
        fflush(stdout)

        // Test 1: Linear code encode/decode consistency
        print("  Starting linear code test...")
        fflush(stdout)
        let code = LinearCode(messageLength: 8, rateInverse: 4, seed: 0xBEEF)
        var rng: UInt64 = 0xDEAD_BEEF_CAFE
        var msg = [Fr](repeating: Fr.zero, count: 8)
        for i in 0..<8 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            msg[i] = frFromInt(rng >> 32)
        }
        let codeword = code.encode(msg)
        let systematic = Array(codeword.prefix(8))
        var sysMatch = true
        for i in 0..<8 {
            if frToInt(systematic[i]) != frToInt(msg[i]) { sysMatch = false; break }
        }
        print("  Linear code systematic property: \(sysMatch ? "PASS" : "FAIL")")
        print("  Codeword length: \(codeword.count) (message: \(code.messageLength), redundancy: \(code.redundancyLength))")
        fflush(stdout)

        // Test 2: Tensor product computation
        print("  Starting tensor test...")
        fflush(stdout)
        let point2 = [frFromInt(3), frFromInt(5)]
        let tensor = engine.computeTensor(point2)
        // tensor should be: [(1-3)(1-5), 3*(1-5), (1-3)*5, 3*5] = [-2*-4, -12, -10, 15]
        // In Fr arithmetic (mod p)
        // tensor[i] = prod_j (if bit j of i is 1: z_j, else 1-z_j)
        // With z0=3, z1=5:
        // tensor[0] = (1-z0)(1-z1), tensor[1] = (1-z0)*z1, tensor[2] = z0*(1-z1), tensor[3] = z0*z1
        let expected0 = frMul(frSub(Fr.one, frFromInt(3)), frSub(Fr.one, frFromInt(5)))
        let expected1 = frMul(frSub(Fr.one, frFromInt(3)), frFromInt(5))
        let expected2 = frMul(frFromInt(3), frSub(Fr.one, frFromInt(5)))
        let expected3 = frMul(frFromInt(3), frFromInt(5))
        let tensorOK = frToInt(tensor[0]) == frToInt(expected0) &&
                        frToInt(tensor[1]) == frToInt(expected1) &&
                        frToInt(tensor[2]) == frToInt(expected2) &&
                        frToInt(tensor[3]) == frToInt(expected3)
        print("  Tensor product (2-var): \(tensorOK ? "PASS" : "FAIL")")

        // Test 3: Commit -> Open -> Verify round-trip
        print("  Starting commit-open-verify tests...")
        fflush(stdout)
        for logN in [8, 10, 12] {
            print("    Testing 2^\(logN)...")
            fflush(stdout)
            let n = 1 << logN
            var evals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = frFromInt(rng >> 32)
            }

            var point = [Fr]()
            for _ in 0..<logN {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                point.append(frFromInt(rng >> 32))
            }

            print("      committing...")
            fflush(stdout)
            let commitment = try engine.commit(evaluations: evals)
            print("      opening...")
            fflush(stdout)
            let proof = try engine.open(evaluations: evals, point: point, commitment: commitment)
            print("      evaluating...")
            fflush(stdout)

            // Compute expected value using CPU multilinear evaluation
            let expectedValue = BrakedownEngine.cpuEvaluate(evaluations: evals, point: point)
            print("      verifying...")
            fflush(stdout)

            let verified = engine.verify(
                commitment: commitment,
                point: point,
                value: expectedValue,
                proof: proof
            )
            print("  Commit-Open-Verify 2^\(logN): \(verified ? "PASS" : "FAIL") " +
                  "(matrix: \(commitment.numRows)x\(commitment.numCols), encoded: \(commitment.numEncodedCols) cols)")
        }

        // Test 4: Verify rejects wrong value
        let smallN = 1 << 8
        var smallEvals = [Fr](repeating: Fr.zero, count: smallN)
        for i in 0..<smallN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            smallEvals[i] = frFromInt(rng >> 32)
        }
        var smallPoint = [Fr]()
        for _ in 0..<8 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            smallPoint.append(frFromInt(rng >> 32))
        }
        let smallCommit = try engine.commit(evaluations: smallEvals)
        let smallProof = try engine.open(evaluations: smallEvals, point: smallPoint, commitment: smallCommit)
        let wrongValue = frFromInt(12345)
        let rejectWrong = !engine.verify(
            commitment: smallCommit,
            point: smallPoint,
            value: wrongValue,
            proof: smallProof
        )
        print("  Reject wrong value: \(rejectWrong ? "PASS" : "FAIL")")

        // --- Performance Benchmarks ---
        print("\n--- Performance benchmarks ---")
        print(String(format: "  %-10s  %-12s  %-12s  %-12s  %-10s", "Size", "Commit", "Open", "Verify", "Total"))

        let benchSizes = CommandLine.arguments.contains("--quick") ? [10, 14] : [10, 14, 18, 20]

        for logN in benchSizes {
            let n = 1 << logN
            var evals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = frFromInt(rng >> 32)
            }

            var point = [Fr]()
            for _ in 0..<logN {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                point.append(frFromInt(rng >> 32))
            }

            // Warmup
            let warmCommit = try engine.commit(evaluations: evals)
            let _ = try engine.open(evaluations: evals, point: point, commitment: warmCommit)

            let runs = 3
            var commitTimes = [Double]()
            var openTimes = [Double]()
            var verifyTimes = [Double]()

            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                let commit = try engine.commit(evaluations: evals)
                let t1 = CFAbsoluteTimeGetCurrent()
                let proof = try engine.open(evaluations: evals, point: point, commitment: commit)
                let t2 = CFAbsoluteTimeGetCurrent()

                let expectedVal = BrakedownEngine.cpuEvaluate(evaluations: evals, point: point)
                let _ = engine.verify(commitment: commit, point: point, value: expectedVal, proof: proof)
                let t3 = CFAbsoluteTimeGetCurrent()

                commitTimes.append((t1 - t0) * 1000)
                openTimes.append((t2 - t1) * 1000)
                verifyTimes.append((t3 - t2) * 1000)
            }

            commitTimes.sort()
            openTimes.sort()
            verifyTimes.sort()
            let commitMs = commitTimes[runs / 2]
            let openMs = openTimes[runs / 2]
            let verifyMs = verifyTimes[runs / 2]
            let totalMs = commitMs + openMs + verifyMs

            print(String(format: "  2^%-8d  %8.2fms   %8.2fms   %8.2fms   %8.2fms",
                         logN, commitMs, openMs, verifyMs, totalMs))
        }

        print("\nNote: Brakedown uses NO NTT — commit is pure matrix-vector multiply + Merkle hash.")
        print("This makes it ideal for GPU acceleration without NTT-friendly field requirements.")

    } catch {
        print("Error: \(error)")
    }
}
