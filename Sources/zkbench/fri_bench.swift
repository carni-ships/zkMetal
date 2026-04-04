// FRI Benchmark — folding performance and correctness
import zkMetal
import Foundation

public func runFRIBench() {
    print("=== FRI Benchmark ===")

    do {
        let engine = try FRIEngine()

        // Correctness: single fold, verify GPU == CPU
        print("\n--- Correctness verification ---")
        let testLogN = 10
        let testN = 1 << testLogN
        var testEvals = [Fr](repeating: Fr.zero, count: testN)
        var rng: UInt64 = 0xDEAD_BEEF
        for i in 0..<testN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testEvals[i] = frFromInt(rng >> 32)
        }
        let beta = frFromInt(42)

        let gpuFolded = try engine.fold(evals: testEvals, beta: beta)
        let cpuFolded = FRIEngine.cpuFold(evals: testEvals, beta: beta, logN: testLogN)

        var correct = true
        for i in 0..<gpuFolded.count {
            if frToInt(gpuFolded[i]) != frToInt(cpuFolded[i]) {
                print("  MISMATCH at \(i)")
                correct = false
                break
            }
        }
        print("  Single fold: \(correct ? "PASS" : "FAIL")")

        // Fold-by-4 correctness: verify GPU fold4 == CPU fold4
        let test4LogN = 12
        let test4N = 1 << test4LogN
        var test4Evals = [Fr](repeating: Fr.zero, count: test4N)
        for i in 0..<test4N {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            test4Evals[i] = frFromInt(rng >> 32)
        }
        let beta4 = frFromInt(99)
        let gpuFolded4 = try engine.fold4(evals: test4Evals, beta: beta4)
        let cpuFolded4 = FRIEngine.cpuFold4(evals: test4Evals, beta: beta4, logN: test4LogN)
        var correct4 = gpuFolded4.count == cpuFolded4.count
        if correct4 {
            for i in 0..<gpuFolded4.count {
                if frToInt(gpuFolded4[i]) != frToInt(cpuFolded4[i]) {
                    print("  FOLD4 MISMATCH at \(i)")
                    correct4 = false
                    break
                }
            }
        }
        print("  Fold-by-4: \(correct4 ? "PASS" : "FAIL") (2^\(test4LogN) → \(gpuFolded4.count))")

        // Multi-fold test: fold from 2^16 down to constant
        let multiLogN = 16
        let multiN = 1 << multiLogN
        var multiEvals = [Fr](repeating: Fr.zero, count: multiN)
        for i in 0..<multiN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            multiEvals[i] = frFromInt(rng >> 32)
        }

        var betas = [Fr]()
        for i in 0..<multiLogN {
            betas.append(frFromInt(UInt64(i + 1) * 7))
        }

        let finalGPU = try engine.multiFold(evals: multiEvals, betas: betas)
        print("  Multi-fold 2^16→1: \(finalGPU.count) element(s) remaining")

        // CPU verification of multi-fold
        var cpuCurrent = multiEvals
        for i in 0..<multiLogN {
            let curLogN = multiLogN - i
            cpuCurrent = FRIEngine.cpuFold(evals: cpuCurrent, beta: betas[i], logN: curLogN)
        }
        let multiCorrect = (finalGPU.count == 1 && cpuCurrent.count == 1 && frToInt(finalGPU[0]) == frToInt(cpuCurrent[0]))
        print("  Multi-fold correctness: \(multiCorrect ? "PASS" : "FAIL")")

        // multiFold4 correctness: fold from 2^16 down using fold-by-4
        let mf4Betas: [Fr] = (0..<(multiLogN / 2)).map { frFromInt(UInt64($0 + 1) * 11) }
        let finalGPU4 = try engine.multiFold4(evals: multiEvals, betas: mf4Betas)
        // CPU verification: sequential cpuFold4
        var cpu4Current = multiEvals
        var cpu4LogN = multiLogN
        for i in 0..<mf4Betas.count {
            cpu4Current = FRIEngine.cpuFold4(evals: cpu4Current, beta: mf4Betas[i], logN: cpu4LogN)
            cpu4LogN -= 2
        }
        let mf4Correct = (finalGPU4.count == cpu4Current.count && finalGPU4.count == 1 && frToInt(finalGPU4[0]) == frToInt(cpu4Current[0]))
        print("  MultiFold4 2^16→1: \(mf4Correct ? "PASS" : "FAIL") (\(finalGPU4.count) element(s))")

        // Performance benchmark: single fold at various sizes
        print("\n--- Fold performance ---")
        let sizes = [14, 16, 18, 20, 22]

        for logN in sizes {
            let n = 1 << logN
            var evals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = frFromInt(rng >> 32)
            }

            // Warmup
            let _ = try engine.fold(evals: evals, beta: beta)

            var times = [Double]()
            for _ in 0..<10 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.fold(evals: evals, beta: beta)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                times.append(elapsed)
            }
            times.sort()
            let median = times[5]
            let elemPerSec = Double(n) / (median / 1000)

            // CPU fold for comparison (skip > 2^16 — too slow)
            var cpuMs: Double = 0
            if logN <= 22 && !skipCPU {
                let cpuT0 = CFAbsoluteTimeGetCurrent()
                let _ = FRIEngine.cpuFold(evals: evals, beta: beta, logN: logN)
                cpuMs = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000
            }

            if cpuMs > 0 {
                print(String(format: "  2^%-2d = %7d | GPU: %7.2fms | CPU: %7.1fms | %.0fx | %.1fM elem/s",
                            logN, n, median, cpuMs, cpuMs / median, elemPerSec / 1e6))
            } else {
                print(String(format: "  2^%-2d = %7d | GPU: %7.2fms | %.1fM elem/s",
                            logN, n, median, elemPerSec / 1e6))
            }
        }

        // Fold-by-4 performance
        print("\n--- Fold-by-4 performance ---")
        for logN in sizes {
            let n = 1 << logN
            var evals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = frFromInt(rng >> 32)
            }

            // Warmup
            let _ = try engine.fold4(evals: evals, beta: beta)

            var times = [Double]()
            for _ in 0..<10 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.fold4(evals: evals, beta: beta)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                times.append(elapsed)
            }
            times.sort()
            let median = times[5]
            let elemPerSec = Double(n) / (median / 1000)
            print(String(format: "  2^%-2d = %7d | GPU fold4: %7.2fms | %.1fM elem/s",
                        logN, n, median, elemPerSec / 1e6))
        }

        // MultiFold4 benchmark
        print("\n--- Full FRI fold-by-4 protocol (fold to constant) ---")
        for startLogN in [16, 18, 20] {
            let n = 1 << startLogN
            var evals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = frFromInt(rng >> 32)
            }
            let numBetas4 = startLogN / 2
            var challenges4 = [Fr]()
            for i in 0..<numBetas4 {
                challenges4.append(frFromInt(UInt64(i + 1) * 19))
            }

            // Warmup
            let _ = try engine.multiFold4(evals: evals, betas: challenges4)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.multiFold4(evals: evals, betas: challenges4)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                times.append(elapsed)
            }
            times.sort()
            let median = times[2]
            print(String(format: "  2^%-2d → 1 (%d fold4 steps): %7.2fms",
                        startLogN, numBetas4, median))
        }

        // FRI commit → query → verify round-trip
        print("\n--- FRI Proof Protocol (commit → query → verify) ---")
        do {
            let protoLogN = 14
            let protoN = 1 << protoLogN
            var protoEvals = [Fr](repeating: Fr.zero, count: protoN)
            for i in 0..<protoN {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                protoEvals[i] = frFromInt(rng >> 32)
            }
            var protoBetas = [Fr]()
            for i in 0..<protoLogN {
                protoBetas.append(frFromInt(UInt64(i + 1) * 17))
            }

            let commitment = try engine.commitPhase(evals: protoEvals, betas: protoBetas)
            print("  Commit: \(commitment.layers.count) layers, \(commitment.roots.count) roots")

            let queryIndices: [UInt32] = [0, 42, 1000, UInt32(protoN / 2 - 1)]
            let queries = try engine.queryPhase(commitment: commitment, queryIndices: queryIndices)
            print("  Query: \(queries.count) proofs extracted")

            let verified = engine.verify(commitment: commitment, queries: queries)
            print("  Verify: \(verified ? "PASS" : "FAIL")")

            // Benchmark commit phase
            let _ = try engine.commitPhase(evals: protoEvals, betas: protoBetas)  // warmup
            engine.profileCommit = true
            let _ = try engine.commitPhase(evals: protoEvals, betas: protoBetas)  // profile run
            engine.profileCommit = false
            var commitTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.commitPhase(evals: protoEvals, betas: protoBetas)
                commitTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            commitTimes.sort()
            print(String(format: "  Commit 2^%d: %.1fms (fold + Merkle)", protoLogN, commitTimes[2]))
        }

        // Multi-fold benchmark: full FRI protocol
        print("\n--- Full FRI protocol (fold to constant) ---")
        for startLogN in [16, 18, 20] {
            let n = 1 << startLogN
            var evals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = frFromInt(rng >> 32)
            }
            var challenges = [Fr]()
            for i in 0..<startLogN {
                challenges.append(frFromInt(UInt64(i + 1) * 13))
            }

            // Warmup
            let _ = try engine.multiFold(evals: evals, betas: challenges)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.multiFold(evals: evals, betas: challenges)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                times.append(elapsed)
            }
            times.sort()
            let median = times[2]
            print(String(format: "  2^%-2d → 1 (%d folds): %7.2fms",
                        startLogN, startLogN, median))
        }

    } catch {
        print("Error: \(error)")
    }
}
