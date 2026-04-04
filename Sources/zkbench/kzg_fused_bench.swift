// Fused Quotient Accumulation Benchmark for Shplonk-style KZG Multi-Point Openings
// Compares sequential (N divideByLinear + CPU accum) vs fused (single GPU pass)
import zkMetal
import Foundation

public func runKZGFusedBench() {
    print("=== Fused Quotient Accumulation Benchmark (Shplonk-style KZG) ===")

    do {
        // Setup: BN254 generator, test SRS
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let generator = PointAffine(x: gx, y: gy)

        let secretLimbs: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]
        let srsSize = 4096

        let srs = KZGEngine.generateTestSRS(secret: secretLimbs, size: srsSize, generator: generator)
        let engine = try KZGEngine(srs: srs)

        // --- Correctness Test ---
        print("\n--- Correctness: fused vs sequential produce same proof ---")

        // Test with 4 polynomials at different points
        let polys: [[Fr]] = [
            [frFromInt(1), frFromInt(2), frFromInt(3)],
            [frFromInt(5), frFromInt(7)],
            [frFromInt(10), frFromInt(0), frFromInt(1), frFromInt(4)],
            [frFromInt(3), frFromInt(3), frFromInt(3), frFromInt(3)],
        ]
        let points = [frFromInt(5), frFromInt(7), frFromInt(3), frFromInt(11)]
        let gamma = frFromInt(17)

        let seqResult = try engine.batchOpenMultiPoint(polynomials: polys, points: points, gamma: gamma)
        let fusedResult = try engine.batchOpenMultiPointFused(polynomials: polys, points: points, gamma: gamma)

        // Compare evaluations
        var evalsMatch = true
        for i in 0..<polys.count {
            let s = frToInt(seqResult.evaluations[i])
            let f = frToInt(fusedResult.evaluations[i])
            if s != f {
                print("  [FAIL] Evaluation mismatch at poly \(i): seq=\(s) fused=\(f)")
                evalsMatch = false
            }
        }
        if evalsMatch {
            print("  [pass] All evaluations match")
        }

        // Compare proof points
        let seqAffine = batchToAffine([seqResult.proof])
        let fusedAffine = batchToAffine([fusedResult.proof])
        let proofMatch = fpToInt(seqAffine[0].x) == fpToInt(fusedAffine[0].x) &&
                         fpToInt(seqAffine[0].y) == fpToInt(fusedAffine[0].y)
        if proofMatch {
            print("  [pass] Proof points match (sequential == fused)")
        } else {
            print("  [FAIL] Proof points differ!")
            print("    seq.x  = \(fpToInt(seqAffine[0].x))")
            print("    fused.x = \(fpToInt(fusedAffine[0].x))")
            print("    seq.y  = \(fpToInt(seqAffine[0].y))")
            print("    fused.y = \(fpToInt(fusedAffine[0].y))")
        }

        // Correctness test with larger random polynomials
        print("\n--- Correctness: larger random polynomials ---")
        for numPolys in [8, 16] {
            let polyDeg = 256
            var rng: UInt64 = 0xABCD_1234_0000 &+ UInt64(numPolys)
            var testPolys = [[Fr]]()
            var testPoints = [Fr]()
            for i in 0..<numPolys {
                var coeffs = [Fr](repeating: Fr.zero, count: polyDeg)
                for j in 0..<polyDeg {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    coeffs[j] = frFromInt(rng >> 32)
                }
                testPolys.append(coeffs)
                testPoints.append(frFromInt(UInt64(100 + i)))
            }

            let testGamma = frFromInt(137)
            let seqR = try engine.batchOpenMultiPoint(polynomials: testPolys, points: testPoints, gamma: testGamma)
            let fusR = try engine.batchOpenMultiPointFused(polynomials: testPolys, points: testPoints, gamma: testGamma)

            let sA = batchToAffine([seqR.proof])
            let fA = batchToAffine([fusR.proof])
            let match = fpToInt(sA[0].x) == fpToInt(fA[0].x) && fpToInt(sA[0].y) == fpToInt(fA[0].y)
            print("  N=\(numPolys) deg=\(polyDeg): \(match ? "[pass]" : "[FAIL]") proofs match")
        }

        // --- Performance Benchmarks ---
        print("\n--- Performance: sequential vs fused multi-point batch open ---")

        for numPolys in [4, 8, 16, 32] {
            let polyDeg = min(1024, srsSize)

            // Generate random polynomials
            var rng: UInt64 = 0xCAFE_BABE_0000 &+ UInt64(numPolys)
            var testPolys = [[Fr]]()
            var testPoints = [Fr]()
            for i in 0..<numPolys {
                var coeffs = [Fr](repeating: Fr.zero, count: polyDeg)
                for j in 0..<polyDeg {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    coeffs[j] = frFromInt(rng >> 32)
                }
                testPolys.append(coeffs)
                testPoints.append(frFromInt(UInt64(100 + i)))
            }

            let testGamma = frFromInt(137)

            // Warmup
            let _ = try engine.batchOpenMultiPoint(polynomials: testPolys, points: testPoints, gamma: testGamma)
            let _ = try engine.batchOpenMultiPointFused(polynomials: testPolys, points: testPoints, gamma: testGamma)

            // Time sequential multi-point batch open
            var seqTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.batchOpenMultiPoint(polynomials: testPolys, points: testPoints, gamma: testGamma)
                seqTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            seqTimes.sort()
            let seqMedian = seqTimes[2]

            // Time fused multi-point batch open
            var fusedTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.batchOpenMultiPointFused(polynomials: testPolys, points: testPoints, gamma: testGamma)
                fusedTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            fusedTimes.sort()
            let fusedMedian = fusedTimes[2]

            let speedup = seqMedian / fusedMedian
            print(String(format: "  N=%-2d deg=%d | sequential: %6.1fms | fused: %6.1fms | speedup: %.2fx",
                        numPolys, polyDeg, seqMedian, fusedMedian, speedup))
        }

        // --- Larger polynomial degrees ---
        print("\n--- Performance: scaling with polynomial degree ---")
        let numPolys = 8
        for logDeg in [8, 10, 11] {
            let polyDeg = 1 << logDeg
            if polyDeg > srsSize { continue }

            var rng: UInt64 = 0xDEAD_0000 &+ UInt64(logDeg)
            var testPolys = [[Fr]]()
            var testPoints = [Fr]()
            for i in 0..<numPolys {
                var coeffs = [Fr](repeating: Fr.zero, count: polyDeg)
                for j in 0..<polyDeg {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    coeffs[j] = frFromInt(rng >> 32)
                }
                testPolys.append(coeffs)
                testPoints.append(frFromInt(UInt64(200 + i)))
            }

            let testGamma = frFromInt(31)

            // Warmup
            let _ = try engine.batchOpenMultiPoint(polynomials: testPolys, points: testPoints, gamma: testGamma)
            let _ = try engine.batchOpenMultiPointFused(polynomials: testPolys, points: testPoints, gamma: testGamma)

            var seqTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.batchOpenMultiPoint(polynomials: testPolys, points: testPoints, gamma: testGamma)
                seqTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            seqTimes.sort()

            var fusedTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.batchOpenMultiPointFused(polynomials: testPolys, points: testPoints, gamma: testGamma)
                fusedTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            fusedTimes.sort()

            let speedup = seqTimes[2] / fusedTimes[2]
            print(String(format: "  N=%d deg=2^%d (%d) | sequential: %6.1fms | fused: %6.1fms | speedup: %.2fx",
                        numPolys, logDeg, polyDeg, seqTimes[2], fusedTimes[2], speedup))
        }

    } catch {
        print("  [FAIL] error: \(error)")
    }

    print("\nFused quotient accumulation benchmark complete.")
}
