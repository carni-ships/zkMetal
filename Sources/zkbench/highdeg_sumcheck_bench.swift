// High-Degree Sumcheck Benchmark — degree-2, 4, 8, 16, 32
import zkMetal
import Foundation

public func runHighDegSumcheckBench() {
    print("=== High-Degree Sumcheck Benchmark ===")

    do {
        let engine = try SumcheckEngine()

        // --- Correctness verification ---
        print("\n--- Correctness verification ---")
        let testNumVars = 10
        let testN = 1 << testNumVars
        var rng: UInt64 = 0xCAFE_BABE

        // Test with degree 2 (k=2 polys), 4, 8, 16
        let testDegrees = [2, 4, 8, 16]

        for degree in testDegrees {
            // Generate k random polynomials
            var polys = [[Fr]]()
            for _ in 0..<degree {
                var p = [Fr](repeating: Fr.zero, count: testN)
                for i in 0..<testN {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    p[i] = frFromInt(rng >> 32)
                }
                polys.append(p)
            }

            // Generate challenges
            var challenges = [Fr]()
            for i in 0..<testNumVars {
                challenges.append(frFromInt(UInt64(i + 1) * 13))
            }

            // GPU
            let (gpuRounds, gpuFinal) = try engine.proveHighDegree(
                polynomials: polys, challenges: challenges)

            // CPU reference
            let (cpuRounds, cpuFinal) = SumcheckEngine.cpuFullHighDegree(
                polynomials: polys, challenges: challenges)

            // Compare rounds
            var roundsMatch = true
            for r in 0..<testNumVars {
                for t in 0..<(degree + 1) {
                    if frToInt(gpuRounds[r][t]) != frToInt(cpuRounds[r][t]) {
                        print("  degree-\(degree): MISMATCH round \(r) eval point \(t)")
                        roundsMatch = false
                        break
                    }
                }
                if !roundsMatch { break }
            }

            // Compare final
            let finalMatch = frToInt(gpuFinal) == frToInt(cpuFinal)

            // Verify protocol
            // Compute claimed sum: sum over hypercube of product of all polys
            var claimedSum = Fr.zero
            for i in 0..<testN {
                var product = Fr.one
                for j in 0..<degree {
                    product = frMul(product, polys[j][i])
                }
                claimedSum = frAdd(claimedSum, product)
            }

            let verified = SumcheckEngine.verifyHighDegree(
                claimedSum: claimedSum,
                rounds: gpuRounds,
                challenges: challenges,
                finalEval: gpuFinal
            )

            let pass = roundsMatch && finalMatch && verified
            print("  degree-\(degree) (\(degree) polys, \(testNumVars) vars): \(pass ? "PASS" : "FAIL")" +
                  (roundsMatch ? "" : " [rounds mismatch]") +
                  (finalMatch ? "" : " [final mismatch]") +
                  (verified ? "" : " [verify fail]"))
        }

        // --- Performance benchmark ---
        print("\n--- Performance ---")
        let degrees = [2, 4, 8, 16, 32]
        let sizes = [12, 14, 16, 18, 20]

        for degree in degrees {
            print("\n  degree-\(degree) (\(degree) polys):")
            for numVars in sizes {
                let n = 1 << numVars

                // Generate random polynomials
                var polys = [[Fr]]()
                for _ in 0..<degree {
                    var p = [Fr](repeating: Fr.zero, count: n)
                    for i in 0..<n {
                        rng = rng &* 6364136223846793005 &+ 1442695040888963407
                        p[i] = frFromInt(rng >> 32)
                    }
                    polys.append(p)
                }

                var chals = [Fr]()
                for i in 0..<numVars { chals.append(frFromInt(UInt64(i + 1))) }

                // Warmup
                let _ = try engine.proveHighDegree(polynomials: polys, challenges: chals)

                var times = [Double]()
                for _ in 0..<5 {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let _ = try engine.proveHighDegree(polynomials: polys, challenges: chals)
                    let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                    times.append(elapsed)
                }
                times.sort()
                let median = times[2]

                // CPU reference timing (only for small sizes)
                var cpuMs: Double = 0
                if !skipCPU && numVars <= 14 && degree <= 8 {
                    let cpuT0 = CFAbsoluteTimeGetCurrent()
                    let _ = SumcheckEngine.cpuFullHighDegree(
                        polynomials: polys, challenges: chals)
                    cpuMs = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000
                }

                if cpuMs > 0 {
                    print(String(format: "    %d vars (2^%d): GPU %8.2fms | CPU %8.1fms | %.0fx",
                                numVars, numVars, median, cpuMs, cpuMs / median))
                } else {
                    print(String(format: "    %d vars (2^%d): GPU %8.2fms",
                                numVars, numVars, median))
                }
            }
        }

    } catch {
        print("Error: \(error)")
    }
}
