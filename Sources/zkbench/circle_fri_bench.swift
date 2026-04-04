// Circle FRI Benchmark — folding performance and correctness over Mersenne31
import zkMetal
import Foundation

public func runCircleFRIBench() {
    print("=== Circle FRI Benchmark (Mersenne31) ===")

    do {
        let engine = try CircleFRIEngine()

        // --- Correctness: single y-fold, GPU vs CPU ---
        print("\n--- Correctness verification ---")
        let testLogN = 10
        let testN = 1 << testLogN
        var rng: UInt64 = 0xCAFE_BABE
        var testEvals = [M31](repeating: M31.zero, count: testN)
        for i in 0..<testN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testEvals[i] = M31(v: UInt32(rng >> 33))  // ensure < p
        }
        let alpha0 = M31(v: 42)

        let gpuFolded = try engine.fold(evals: testEvals, alpha: alpha0, logN: testLogN,
                                         isFirstFold: true)
        let cpuFolded = CircleFRIEngine.cpuFold(evals: testEvals, alpha: alpha0, logN: testLogN,
                                                  isFirstFold: true)

        var correct = gpuFolded.count == cpuFolded.count
        if correct {
            for i in 0..<gpuFolded.count {
                if gpuFolded[i].v != cpuFolded[i].v {
                    print("  Y-FOLD MISMATCH at \(i): GPU=\(gpuFolded[i].v) CPU=\(cpuFolded[i].v)")
                    correct = false
                    break
                }
            }
        }
        print("  Single y-fold (2^\(testLogN)): \(correct ? "PASS" : "FAIL")")

        // --- Correctness: multi-fold (y + x folds), GPU vs CPU ---
        let multiLogN = 12
        let multiN = 1 << multiLogN
        var multiEvals = [M31](repeating: M31.zero, count: multiN)
        for i in 0..<multiN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            multiEvals[i] = M31(v: UInt32(rng >> 33))
        }

        let numRounds = 4
        var alphas = [M31]()
        for r in 0..<numRounds {
            alphas.append(M31(v: UInt32(17 + r * 13)))
        }

        let gpuMulti = try engine.multiFold(evals: multiEvals, alphas: alphas)
        let cpuMulti = CircleFRIEngine.cpuMultiFold(evals: multiEvals, alphas: alphas, logN: multiLogN)

        var multiCorrect = gpuMulti.count == cpuMulti.count
        if multiCorrect {
            for i in 0..<gpuMulti.count {
                if gpuMulti[i].v != cpuMulti[i].v {
                    print("  MULTI-FOLD MISMATCH at \(i): GPU=\(gpuMulti[i].v) CPU=\(cpuMulti[i].v)")
                    multiCorrect = false
                    break
                }
            }
        }
        print("  Multi-fold (2^\(multiLogN), \(numRounds) rounds -> \(gpuMulti.count)): \(multiCorrect ? "PASS" : "FAIL")")

        // --- Correctness: commit + verify ---
        let commitLogN = 10
        let commitN = 1 << commitLogN
        var commitEvals = [M31](repeating: M31.zero, count: commitN)
        for i in 0..<commitN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            commitEvals[i] = M31(v: UInt32(rng >> 33))
        }

        let commitRounds = commitLogN - 1
        var commitAlphas = [M31]()
        for r in 0..<commitRounds {
            commitAlphas.append(M31(v: UInt32(7 + r * 11)))
        }

        let commitment = try engine.commitPhase(evals: commitEvals, alphas: commitAlphas)
        print("  Commit phase (2^\(commitLogN), \(commitRounds) rounds): layers=\(commitment.layers.count), roots=\(commitment.roots.count)")

        // Query phase
        let queryIndices: [UInt32] = [0, 1, UInt32(commitN / 4), UInt32(commitN / 2 - 1)]
        let queries = engine.queryPhase(commitment: commitment, queryIndices: queryIndices)
        print("  Query phase: \(queries.count) queries generated")

        // Verify
        let verified = engine.verify(commitment: commitment, queries: queries)
        print("  Verify: \(verified ? "PASS" : "FAIL")")

        // --- Performance benchmarks ---
        print("\n--- Performance ---")
        let warmup = 2
        let iters = 5

        for logN in [14, 18, 20] {
            let n = 1 << logN
            var evals = [M31](repeating: M31.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = M31(v: UInt32(rng >> 33))
            }

            // Single y-fold benchmark
            let alpha = M31(v: 42)
            for _ in 0..<warmup {
                _ = try engine.fold(evals: evals, alpha: alpha, logN: logN, isFirstFold: true)
            }
            var t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iters {
                _ = try engine.fold(evals: evals, alpha: alpha, logN: logN, isFirstFold: true)
            }
            let singleMs = (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
            print(String(format: "  Single fold 2^%-2d: %7.2fms (%d elements)", logN, singleMs, n))

            // Multi-fold benchmark (fold down to constant)
            let rounds = logN - 1
            var betas = [M31]()
            for r in 0..<rounds {
                betas.append(M31(v: UInt32(r + 1)))
            }
            for _ in 0..<warmup {
                _ = try engine.multiFold(evals: evals, alphas: betas)
            }
            t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iters {
                _ = try engine.multiFold(evals: evals, alphas: betas)
            }
            let multiMs = (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
            let finalSize = 1 << (logN - rounds)
            print(String(format: "  Multi-fold 2^%-2d (%d rounds -> %d): %7.2fms", logN, rounds, finalSize, multiMs))
        }

        // Commit phase benchmark
        print("\n--- Commit Phase ---")
        for logN in [14, 18, 20] {
            let n = 1 << logN
            var evals = [M31](repeating: M31.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = M31(v: UInt32(rng >> 33))
            }
            let rounds = logN - 1
            var betas = [M31]()
            for r in 0..<rounds { betas.append(M31(v: UInt32(r + 1))) }

            for _ in 0..<warmup {
                _ = try engine.commitPhase(evals: evals, alphas: betas)
            }

            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iters {
                _ = try engine.commitPhase(evals: evals, alphas: betas)
            }
            let ms = (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
            print(String(format: "  Commit 2^%-2d (%d rounds): %7.2fms", logN, rounds, ms))
        }

    } catch {
        print("  ERROR: \(error)")
    }
}
