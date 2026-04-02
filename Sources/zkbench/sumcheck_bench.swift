// Sumcheck Benchmark — reduce and round polynomial performance
import zkMetal
import Foundation

public func runSumcheckBench() {
    print("=== Sumcheck Benchmark ===")

    do {
        let engine = try SumcheckEngine()

        // Correctness: single reduce step
        print("\n--- Correctness verification ---")
        let testNumVars = 10
        let testN = 1 << testNumVars
        var testEvals = [Fr](repeating: Fr.zero, count: testN)
        var rng: UInt64 = 0xDEAD_BEEF
        for i in 0..<testN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testEvals[i] = frFromInt(rng >> 32)
        }
        let challenge = frFromInt(42)

        let gpuReduced = try engine.reduce(evals: testEvals, challenge: challenge)
        let cpuReduced = SumcheckEngine.cpuReduce(evals: testEvals, challenge: challenge)

        var reduceCorrect = true
        for i in 0..<gpuReduced.count {
            if frToInt(gpuReduced[i]) != frToInt(cpuReduced[i]) {
                print("  Reduce MISMATCH at \(i)")
                reduceCorrect = false
                break
            }
        }
        print("  Reduce: \(reduceCorrect ? "PASS" : "FAIL")")

        // Round polynomial correctness
        let evalsBuf = engine.device.makeBuffer(
            length: testN * MemoryLayout<Fr>.stride, options: .storageModeShared)!
        testEvals.withUnsafeBytes { src in
            memcpy(evalsBuf.contents(), src.baseAddress!, testN * MemoryLayout<Fr>.stride)
        }

        let (gpuS0, gpuS1, gpuS2) = try engine.computeRoundPoly(evals: evalsBuf, n: testN)
        let (cpuS0, cpuS1, cpuS2) = SumcheckEngine.cpuRoundPoly(evals: testEvals)

        let roundCorrect = frToInt(gpuS0) == frToInt(cpuS0) &&
                           frToInt(gpuS1) == frToInt(cpuS1) &&
                           frToInt(gpuS2) == frToInt(cpuS2)
        print("  Round poly: \(roundCorrect ? "PASS" : "FAIL")")

        // Sumcheck identity: S(0) + S(1) should equal the claimed sum
        let claimed = frToInt(frAdd(gpuS0, gpuS1))
        var totalSum = Fr.zero
        for e in testEvals { totalSum = frAdd(totalSum, e) }
        let sumCorrect = claimed == frToInt(totalSum)
        print("  S(0)+S(1) = sum: \(sumCorrect ? "PASS" : "FAIL")")

        // Full sumcheck protocol
        let fullNumVars = 16
        let fullN = 1 << fullNumVars
        var fullEvals = [Fr](repeating: Fr.zero, count: fullN)
        for i in 0..<fullN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            fullEvals[i] = frFromInt(rng >> 32)
        }
        var challenges = [Fr]()
        for i in 0..<fullNumVars {
            challenges.append(frFromInt(UInt64(i + 1) * 7))
        }

        let (rounds, finalEval) = try engine.fullSumcheck(evals: fullEvals, challenges: challenges)
        print("  Full sumcheck (\(fullNumVars) vars): \(rounds.count) rounds, final eval computed")

        // Verify: for each round, S(0) + S(1) should equal running sum
        // (which starts as sum of all evals and gets updated each round)
        var runningEvals = fullEvals
        var protocolCorrect = true
        for i in 0..<fullNumVars {
            let (s0, s1, _) = rounds[i]
            var expected = Fr.zero
            for e in runningEvals { expected = frAdd(expected, e) }
            let roundSum = frAdd(s0, s1)
            if frToInt(roundSum) != frToInt(expected) {
                print("  Protocol FAIL at round \(i)")
                protocolCorrect = false
                break
            }
            runningEvals = SumcheckEngine.cpuReduce(evals: runningEvals, challenge: challenges[i])
        }
        if protocolCorrect {
            print("  Protocol verification: PASS")
        }

        // Performance benchmark
        print("\n--- Performance ---")
        let sizes = [14, 16, 18, 20, 22]

        for numVars in sizes {
            let n = 1 << numVars
            var evals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = frFromInt(rng >> 32)
            }
            var chals = [Fr]()
            for i in 0..<numVars { chals.append(frFromInt(UInt64(i + 1))) }

            // Warmup
            let _ = try engine.fullSumcheck(evals: evals, challenges: chals)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.fullSumcheck(evals: evals, challenges: chals)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                times.append(elapsed)
            }
            times.sort()
            let median = times[2]

            // CPU sumcheck for comparison (skip > 2^16 — too slow)
            var cpuMs: Double = 0
            if numVars <= 22 && !skipCPU {
                var cpuEvals = evals
                let cpuT0 = CFAbsoluteTimeGetCurrent()
                for i in 0..<numVars {
                    let _ = SumcheckEngine.cpuRoundPoly(evals: cpuEvals)
                    cpuEvals = SumcheckEngine.cpuReduce(evals: cpuEvals, challenge: chals[i])
                }
                cpuMs = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000
            }

            if cpuMs > 0 {
                print(String(format: "  %d vars (2^%d evals): GPU %7.2fms | CPU %7.1fms | %.0fx",
                            numVars, numVars, median, cpuMs, cpuMs / median))
            } else {
                print(String(format: "  %d vars (2^%d evals): GPU %7.2fms",
                            numVars, numVars, median))
            }
        }

    } catch {
        print("Error: \(error)")
    }
}
