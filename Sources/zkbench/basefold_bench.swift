// Basefold Polynomial Commitment Benchmark and Correctness Tests
import zkMetal
import Foundation

public func runBasefoldBench() {
    print("=== Basefold Polynomial Commitment Benchmark ===")

    do {
        let engine = try BasefoldEngine()

        // --- Correctness Tests ---
        print("\n--- Correctness verification ---")

        // Test 1: GPU fold matches CPU fold
        let testLogN = 10
        let testN = 1 << testLogN
        var rng: UInt64 = 0xDEAD_BEEF
        var testEvals = [Fr](repeating: Fr.zero, count: testN)
        for i in 0..<testN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testEvals[i] = frFromInt(rng >> 32)
        }
        let alpha = frFromInt(42)

        let gpuFolded = try engine.fold(evals: testEvals, alpha: alpha)
        let cpuFolded = BasefoldEngine.cpuFold(evals: testEvals, alpha: alpha)

        var foldCorrect = gpuFolded.count == cpuFolded.count
        if foldCorrect {
            for i in 0..<gpuFolded.count {
                if frToInt(gpuFolded[i]) != frToInt(cpuFolded[i]) {
                    foldCorrect = false
                    break
                }
            }
        }
        print("  Single fold (2^\(testLogN)): \(foldCorrect ? "PASS" : "FAIL")")

        // Test 2: Multi-fold matches CPU sequential fold
        let multiLogN = 14
        let multiN = 1 << multiLogN
        var multiEvals = [Fr](repeating: Fr.zero, count: multiN)
        for i in 0..<multiN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            multiEvals[i] = frFromInt(rng >> 32)
        }
        var challenges = [Fr]()
        for i in 0..<multiLogN {
            challenges.append(frFromInt(UInt64(i + 1) * 7))
        }

        let gpuResult = try engine.multiFold(evals: multiEvals, challenges: challenges)
        let cpuResult = BasefoldEngine.cpuEvaluate(evals: multiEvals, point: challenges)
        let multiFoldCorrect = gpuResult.count == 1 && frToInt(gpuResult[0]) == frToInt(cpuResult)
        print("  Multi-fold 2^\(multiLogN)->1: \(multiFoldCorrect ? "PASS" : "FAIL")")

        // Test 3: Commit -> Open -> Verify round-trip
        let protoLogN = 10
        let protoN = 1 << protoLogN
        var protoEvals = [Fr](repeating: Fr.zero, count: protoN)
        for i in 0..<protoN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            protoEvals[i] = frFromInt(rng >> 32)
        }
        var protoPoint = [Fr]()
        for i in 0..<protoLogN {
            protoPoint.append(frFromInt(UInt64(i + 1) * 13))
        }

        let commitment = try engine.commit(evaluations: protoEvals)
        let expectedValue = BasefoldEngine.cpuEvaluate(evals: protoEvals, point: protoPoint)
        let proof = try engine.open(commitment: commitment, point: protoPoint)

        // Check final value matches CPU evaluation
        let evalCorrect = frToInt(proof.finalValue) == frToInt(expectedValue)
        print("  Evaluation correctness: \(evalCorrect ? "PASS" : "FAIL")")

        // Verify proof
        let verified = engine.verify(root: commitment.root, point: protoPoint,
                                     claimedValue: expectedValue, proof: proof)
        print("  Proof verification: \(verified ? "PASS" : "FAIL")")

        // Test 4: Verify rejects wrong evaluation
        let wrongValue = frFromInt(999999)
        let rejectedWrong = !engine.verify(root: commitment.root, point: protoPoint,
                                           claimedValue: wrongValue, proof: proof)
        print("  Reject wrong value: \(rejectedWrong ? "PASS" : "FAIL")")

        // --- Performance: Fold ---
        print("\n--- Fold performance ---")
        let foldSizes = [10, 14, 18, 22]
        for logN in foldSizes {
            let n = 1 << logN
            var evals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = frFromInt(rng >> 32)
            }

            // Warmup
            let _ = try engine.fold(evals: evals, alpha: alpha)

            var times = [Double]()
            for _ in 0..<10 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.fold(evals: evals, alpha: alpha)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[5]
            let elemPerSec = Double(n) / (median / 1000)
            print(String(format: "  2^%-2d = %7d | GPU: %7.2fms | %.1fM elem/s",
                        logN, n, median, elemPerSec / 1e6))
        }

        // --- Performance: Multi-fold (full protocol fold) ---
        print("\n--- Multi-fold to constant ---")
        for startLogN in [14, 18, 22] {
            let n = 1 << startLogN
            var evals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = frFromInt(rng >> 32)
            }
            var chals = [Fr]()
            for i in 0..<startLogN {
                chals.append(frFromInt(UInt64(i + 1) * 11))
            }

            let _ = try engine.multiFold(evals: evals, challenges: chals)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.multiFold(evals: evals, challenges: chals)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            print(String(format: "  2^%-2d -> 1 (%d folds): %7.2fms",
                        startLogN, startLogN, times[2]))
        }

        // --- Performance: Commit + Open + Verify ---
        print("\n--- Commit + Open + Verify ---")
        for logN in [10, 14, 18] {
            let n = 1 << logN
            var evals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = frFromInt(rng >> 32)
            }
            var pt = [Fr]()
            for i in 0..<logN {
                pt.append(frFromInt(UInt64(i + 1) * 17))
            }

            // Warmup
            let warmCommit = try engine.commit(evaluations: evals)
            let _ = try engine.open(commitment: warmCommit, point: pt)

            // Benchmark commit
            var commitTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.commit(evaluations: evals)
                commitTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            commitTimes.sort()

            // Benchmark open
            let comm = try engine.commit(evaluations: evals)
            var openTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.open(commitment: comm, point: pt)
                openTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            openTimes.sort()

            // Benchmark verify
            let prf = try engine.open(commitment: comm, point: pt)
            let expectedVal = BasefoldEngine.cpuEvaluate(evals: evals, point: pt)
            var verifyTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = engine.verify(root: comm.root, point: pt,
                                      claimedValue: expectedVal, proof: prf)
                verifyTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            verifyTimes.sort()

            print(String(format: "  2^%-2d | commit: %7.2fms | open: %7.2fms | verify: %7.2fms | total: %7.2fms",
                        logN, commitTimes[2], openTimes[2], verifyTimes[2],
                        commitTimes[2] + openTimes[2] + verifyTimes[2]))
        }

    } catch {
        print("Error: \(error)")
    }
}
