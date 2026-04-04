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

        // Fold-by-8 correctness: verify GPU fold8 == CPU fold8
        let test8LogN = 12
        let test8N = 1 << test8LogN
        var test8Evals = [Fr](repeating: Fr.zero, count: test8N)
        for i in 0..<test8N {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            test8Evals[i] = frFromInt(rng >> 32)
        }
        let beta8 = frFromInt(77)
        let gpuFolded8 = try engine.fold8(evals: test8Evals, beta: beta8)
        let cpuFolded8 = FRIEngine.cpuFold8(evals: test8Evals, beta: beta8, logN: test8LogN)
        var correct8 = gpuFolded8.count == cpuFolded8.count
        if correct8 {
            for i in 0..<gpuFolded8.count {
                if frToInt(gpuFolded8[i]) != frToInt(cpuFolded8[i]) {
                    print("  FOLD8 MISMATCH at \(i)")
                    correct8 = false
                    break
                }
            }
        }
        print("  Fold-by-8: \(correct8 ? "PASS" : "FAIL") (2^\(test8LogN) → \(gpuFolded8.count))")

        // multiFold8 correctness: fold from 2^18 down using fold-by-8
        let mf8LogN = 18
        let mf8N = 1 << mf8LogN
        var mf8Evals = [Fr](repeating: Fr.zero, count: mf8N)
        for i in 0..<mf8N {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            mf8Evals[i] = frFromInt(rng >> 32)
        }
        // 18 % 3 == 0, so 6 betas needed (all fold-by-8)
        let mf8Betas: [Fr] = (0..<6).map { frFromInt(UInt64($0 + 1) * 13) }
        let finalGPU8 = try engine.multiFold8(evals: mf8Evals, betas: mf8Betas)
        // CPU verification: sequential cpuFold8
        var cpu8Current = mf8Evals
        var cpu8LogN = mf8LogN
        for i in 0..<mf8Betas.count {
            cpu8Current = FRIEngine.cpuFold8(evals: cpu8Current, beta: mf8Betas[i], logN: cpu8LogN)
            cpu8LogN -= 3
        }
        let mf8Correct = (finalGPU8.count == cpu8Current.count && finalGPU8.count == 1 && frToInt(finalGPU8[0]) == frToInt(cpu8Current[0]))
        print("  MultiFold8 2^18→1: \(mf8Correct ? "PASS" : "FAIL") (\(finalGPU8.count) element(s))")

        // MultiFold8 with remainder: logN=16 (16%3=1, one fold-by-2 then 5 fold-by-8)
        let mf8r1LogN = 16
        let mf8r1N = 1 << mf8r1LogN
        var mf8r1Evals = [Fr](repeating: Fr.zero, count: mf8r1N)
        for i in 0..<mf8r1N {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            mf8r1Evals[i] = frFromInt(rng >> 32)
        }
        // 16%3=1: 1 fold-by-2 + 5 fold-by-8 = 6 betas
        let mf8r1Betas: [Fr] = (0..<6).map { frFromInt(UInt64($0 + 1) * 17) }
        let finalGPU8r1 = try engine.multiFold8(evals: mf8r1Evals, betas: mf8r1Betas)
        // CPU: one fold-by-2 then fold-by-8
        var cpu8r1 = mf8r1Evals
        var cpu8r1LogN = mf8r1LogN
        cpu8r1 = FRIEngine.cpuFold(evals: cpu8r1, beta: mf8r1Betas[0], logN: cpu8r1LogN)
        cpu8r1LogN -= 1
        for i in 1..<mf8r1Betas.count {
            cpu8r1 = FRIEngine.cpuFold8(evals: cpu8r1, beta: mf8r1Betas[i], logN: cpu8r1LogN)
            cpu8r1LogN -= 3
        }
        let mf8r1Correct = (finalGPU8r1.count == cpu8r1.count && finalGPU8r1.count == 1 && frToInt(finalGPU8r1[0]) == frToInt(cpu8r1[0]))
        print("  MultiFold8 2^16→1 (rem=1): \(mf8r1Correct ? "PASS" : "FAIL") (\(finalGPU8r1.count) element(s))")

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

        // FRI commit-by-4 → query → verify round-trip
        print("\n--- FRI Proof Protocol fold-by-4 (commit → query → verify) ---")
        do {
            let protoLogN = 14
            let protoN = 1 << protoLogN
            var protoEvals = [Fr](repeating: Fr.zero, count: protoN)
            for i in 0..<protoN {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                protoEvals[i] = frFromInt(rng >> 32)
            }
            // fold-by-4 needs logN/2 betas (7 for logN=14)
            let numBetas4 = protoLogN / 2
            var protoBetas4 = [Fr]()
            for i in 0..<numBetas4 {
                protoBetas4.append(frFromInt(UInt64(i + 1) * 17))
            }

            let commitment4 = try engine.commitPhase4(evals: protoEvals, betas: protoBetas4)
            print("  Commit4: \(commitment4.layers.count) layers, \(commitment4.roots.count) roots")

            let queryIndices: [UInt32] = [0, 42, 1000, UInt32(protoN / 4 - 1)]
            let queries4 = try engine.queryPhase4(commitment: commitment4, queryIndices: queryIndices)
            print("  Query4: \(queries4.count) proofs extracted")

            let verified4 = engine.verify4(commitment: commitment4, queries: queries4)
            print("  Verify4: \(verified4 ? "PASS" : "FAIL")")

            // Benchmark commit phase 4
            let _ = try engine.commitPhase4(evals: protoEvals, betas: protoBetas4)  // warmup
            engine.profileCommit = true
            let _ = try engine.commitPhase4(evals: protoEvals, betas: protoBetas4)  // profile run
            engine.profileCommit = false
            var commitTimes4 = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.commitPhase4(evals: protoEvals, betas: protoBetas4)
                commitTimes4.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            commitTimes4.sort()
            print(String(format: "  Commit4 2^%d: %.1fms (fold + Merkle, %d layers vs %d)",
                        protoLogN, commitTimes4[2], numBetas4 + 1, protoLogN + 1))
        }

        // Fold-by-8 performance
        print("\n--- Fold-by-8 performance ---")
        for logN in sizes {
            let n = 1 << logN
            var evals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = frFromInt(rng >> 32)
            }

            // Warmup
            let _ = try engine.fold8(evals: evals, beta: beta)

            var times = [Double]()
            for _ in 0..<10 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.fold8(evals: evals, beta: beta)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                times.append(elapsed)
            }
            times.sort()
            let median = times[5]
            let elemPerSec = Double(n) / (median / 1000)
            print(String(format: "  2^%-2d = %7d | GPU fold8: %7.2fms | %.1fM elem/s",
                        logN, n, median, elemPerSec / 1e6))
        }

        // FRI commit-by-8 → query → verify round-trip
        print("\n--- FRI Proof Protocol fold-by-8 (commit → query → verify) ---")
        do {
            let protoLogN = 15  // 15%3=0, clean fold-by-8
            let protoN = 1 << protoLogN
            var protoEvals = [Fr](repeating: Fr.zero, count: protoN)
            for i in 0..<protoN {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                protoEvals[i] = frFromInt(rng >> 32)
            }
            let numBetas8 = protoLogN / 3  // 5 for logN=15
            var protoBetas8 = [Fr]()
            for i in 0..<numBetas8 {
                protoBetas8.append(frFromInt(UInt64(i + 1) * 17))
            }

            let commitment8 = try engine.commitPhase8(evals: protoEvals, betas: protoBetas8)
            print("  Commit8: \(commitment8.layers.count) layers, \(commitment8.roots.count) roots")

            let queryIndices: [UInt32] = [0, 42, 1000, UInt32(protoN / 8 - 1)]
            let queries8 = try engine.queryPhase8(commitment: commitment8, queryIndices: queryIndices)
            print("  Query8: \(queries8.count) proofs extracted")

            let verified8 = engine.verify8(commitment: commitment8, queries: queries8)
            print("  Verify8: \(verified8 ? "PASS" : "FAIL")")

            // Benchmark commit phase 8
            let _ = try engine.commitPhase8(evals: protoEvals, betas: protoBetas8)  // warmup
            engine.profileCommit = true
            let _ = try engine.commitPhase8(evals: protoEvals, betas: protoBetas8)  // profile run
            engine.profileCommit = false
            var commitTimes8 = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.commitPhase8(evals: protoEvals, betas: protoBetas8)
                commitTimes8.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            commitTimes8.sort()
            print(String(format: "  Commit8 2^%d: %.1fms (fold + Merkle, %d layers)",
                        protoLogN, commitTimes8[2], numBetas8 + 1))
        }

        // Commit phase comparison at multiple sizes
        print("\n--- Commit phase: fold-by-2 vs fold-by-4 vs fold-by-8 ---")
        for commitLogN in [15, 16, 18, 20] {
            let commitN = 1 << commitLogN
            var commitEvals = [Fr](repeating: Fr.zero, count: commitN)
            for i in 0..<commitN {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                commitEvals[i] = frFromInt(rng >> 32)
            }
            // fold-by-2 betas
            var betas2 = [Fr]()
            for i in 0..<commitLogN { betas2.append(frFromInt(UInt64(i + 1) * 23)) }
            // fold-by-4 betas: oddStart + fold4Count
            let odd4 = (commitLogN % 2 != 0) ? 1 : 0
            let nb4 = odd4 + (commitLogN - odd4) / 2
            var betas4 = [Fr]()
            for i in 0..<nb4 { betas4.append(frFromInt(UInt64(i + 1) * 23)) }
            // fold-by-8 betas: remainder betas + fold8Count
            let rem8 = commitLogN % 3
            let nb8 = (rem8 == 0 ? 0 : 1) + (commitLogN - (rem8 == 0 ? 0 : rem8)) / 3
            var betas8 = [Fr]()
            for i in 0..<nb8 { betas8.append(frFromInt(UInt64(i + 1) * 23)) }

            // Warmup
            let _ = try engine.commitPhase(evals: commitEvals, betas: betas2)
            let _ = try engine.commitPhase4(evals: commitEvals, betas: betas4)
            let _ = try engine.commitPhase8(evals: commitEvals, betas: betas8)

            var times2 = [Double]()
            var times4 = [Double]()
            var times8 = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.commitPhase(evals: commitEvals, betas: betas2)
                times2.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.commitPhase4(evals: commitEvals, betas: betas4)
                times4.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.commitPhase8(evals: commitEvals, betas: betas8)
                times8.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times2.sort()
            times4.sort()
            times8.sort()
            let m2 = times2[2]
            let m4 = times4[2]
            let m8 = times8[2]
            print(String(format: "  2^%-2d | fold-by-2: %7.1fms (%d layers) | fold-by-4: %7.1fms (%d layers) | fold-by-8: %7.1fms (%d layers) | 4/2: %.1fx | 8/2: %.1fx",
                        commitLogN, m2, commitLogN + 1, m4, nb4 + 1, m8, nb8 + 1, m2 / m4, m2 / m8))
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
