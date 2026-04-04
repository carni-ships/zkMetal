// KZG Batch Opening Benchmark and Correctness Tests
import zkMetal
import Foundation

public func runKZGBatchBench() {
    print("=== KZG Batch Opening Benchmark (BN254 G1) ===")

    do {
        // Setup: BN254 generator, test SRS
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let generator = PointAffine(x: gx, y: gy)

        let secretLimbs: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]
        let secret = frFromLimbs(secretLimbs)
        let srsSize = 1024

        let srs = KZGEngine.generateTestSRS(secret: secretLimbs, size: srsSize, generator: generator)
        let engine = try KZGEngine(srs: srs)

        // --- Correctness Tests ---
        print("\n--- Correctness Tests ---")

        // Test 1: Batch open 4 polynomials at the same point
        let polys: [[Fr]] = [
            [frFromInt(1), frFromInt(2), frFromInt(3)],             // 1 + 2x + 3x^2
            [frFromInt(5), frFromInt(7)],                            // 5 + 7x
            [frFromInt(10), frFromInt(0), frFromInt(1), frFromInt(4)], // 10 + x^2 + 4x^3
            [frFromInt(3), frFromInt(3), frFromInt(3), frFromInt(3)], // 3 + 3x + 3x^2 + 3x^3
        ]

        let z = frFromInt(5)
        let gamma = frFromInt(17)

        let batch = try engine.batchOpen(polynomials: polys, point: z, gamma: gamma)

        // Verify evaluations manually:
        // p0(5) = 1 + 10 + 75 = 86
        // p1(5) = 5 + 35 = 40
        // p2(5) = 10 + 0 + 25 + 500 = 535
        // p3(5) = 3 + 15 + 75 + 375 = 468
        let expectedEvals: [UInt64] = [86, 40, 535, 468]
        var evalPass = true
        for i in 0..<4 {
            let got = frToInt(batch.evaluations[i])[0]
            if got != expectedEvals[i] {
                print("  [FAIL] Eval p\(i)(5) = \(got), expected \(expectedEvals[i])")
                evalPass = false
            }
        }
        if evalPass {
            print("  [pass] All 4 evaluations correct")
        }

        // Verify batch proof is non-trivial
        if !pointIsIdentity(batch.proof) {
            print("  [pass] Batch proof is non-trivial")
        } else {
            print("  [FAIL] Batch proof is identity")
        }

        // Verify batch proof is deterministic
        let batch2 = try engine.batchOpen(polynomials: polys, point: z, gamma: gamma)
        let p1a = batchToAffine([batch.proof])
        let p2a = batchToAffine([batch2.proof])
        if fpToInt(p1a[0].x) == fpToInt(p2a[0].x) && fpToInt(p1a[0].y) == fpToInt(p2a[0].y) {
            print("  [pass] Batch proof is deterministic")
        } else {
            print("  [FAIL] Batch proof is not deterministic")
        }

        // Verify via re-open
        let reOpenValid = try engine.verifyBatchByReopen(
            polynomials: polys, point: z, evaluations: batch.evaluations,
            proof: batch.proof, gamma: gamma)
        if reOpenValid {
            print("  [pass] Batch verify-by-reopen accepts valid proof")
        } else {
            print("  [FAIL] Batch verify-by-reopen rejected valid proof")
        }

        // Verify via algebraic check with known secret
        let algebraicValid = engine.batchVerify(
            commitments: batch.commitments, point: z, evaluations: batch.evaluations,
            proof: batch.proof, gamma: gamma, srsSecret: secret)
        if algebraicValid {
            print("  [pass] Batch algebraic verify accepts valid proof")
        } else {
            print("  [FAIL] Batch algebraic verify rejected valid proof")
        }

        // Reject tampered evaluation
        var tamperedEvals = batch.evaluations
        tamperedEvals[0] = frFromInt(999)
        let tamperedValid = engine.batchVerify(
            commitments: batch.commitments, point: z, evaluations: tamperedEvals,
            proof: batch.proof, gamma: gamma, srsSecret: secret)
        if !tamperedValid {
            print("  [pass] Batch verify rejects tampered evaluation")
        } else {
            print("  [FAIL] Batch verify accepted tampered evaluation")
        }

        // Test 2: Multi-point batch opening
        let points = [frFromInt(5), frFromInt(7), frFromInt(3), frFromInt(11)]
        let multiProof = try engine.batchOpenMultiPoint(polynomials: polys, points: points, gamma: gamma)

        // Verify evaluations:
        // p0(5) = 86, p1(7) = 5+49 = 54, p2(3) = 10+0+9+108 = 127, p3(11) = 3+33+363+3993 = 4392
        let expectedMultiEvals: [UInt64] = [86, 54, 127, 4392]
        var multiPass = true
        for i in 0..<4 {
            let got = frToInt(multiProof.evaluations[i])[0]
            if got != expectedMultiEvals[i] {
                print("  [FAIL] Multi-point eval p\(i)(z_\(i)) = \(got), expected \(expectedMultiEvals[i])")
                multiPass = false
            }
        }
        if multiPass {
            print("  [pass] All 4 multi-point evaluations correct")
        }

        // Verify multi-point proof by re-open
        let multiReOpenValid = try engine.verifyMultiPointByReopen(
            polynomials: polys, points: points, evaluations: multiProof.evaluations,
            proof: multiProof.proof, gamma: gamma)
        if multiReOpenValid {
            print("  [pass] Multi-point verify-by-reopen accepts valid proof")
        } else {
            print("  [FAIL] Multi-point verify-by-reopen rejected valid proof")
        }

        // --- Performance Benchmarks ---
        print("\n--- Batch Open: N individual opens vs 1 batch open ---")

        for numPolys in [4, 8, 16, 32] {
            let polyDeg = min(256, srsSize)  // degree of each polynomial

            // Generate random polynomials
            var rng: UInt64 = 0xDEAD_BEEF_0000 &+ UInt64(numPolys)
            var testPolys = [[Fr]]()
            for _ in 0..<numPolys {
                var coeffs = [Fr](repeating: Fr.zero, count: polyDeg)
                for j in 0..<polyDeg {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    coeffs[j] = frFromInt(rng >> 32)
                }
                testPolys.append(coeffs)
            }

            let testZ = frFromInt(42)
            let testGamma = frFromInt(137)

            // Warmup
            for p in testPolys { let _ = try engine.open(p, at: testZ) }
            let _ = try engine.batchOpen(polynomials: testPolys, point: testZ, gamma: testGamma)

            // Time N individual opens
            var individualTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                for p in testPolys {
                    let _ = try engine.open(p, at: testZ)
                }
                individualTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            individualTimes.sort()
            let indivMedian = individualTimes[2]

            // Time 1 batch open
            var batchTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.batchOpen(polynomials: testPolys, point: testZ, gamma: testGamma)
                batchTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            batchTimes.sort()
            let batchMedian = batchTimes[2]

            let speedup = indivMedian / batchMedian
            print(String(format: "  N=%-2d deg=%d | %d individual: %6.1fms | 1 batch: %6.1fms | speedup: %.1fx",
                        numPolys, polyDeg, numPolys, indivMedian, batchMedian, speedup))
        }

        // --- Multi-point benchmark ---
        print("\n--- Multi-Point Batch Open ---")
        for numPolys in [4, 8, 16] {
            let polyDeg = min(256, srsSize)

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

            // Time N individual opens (at different points)
            var individualTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                for i in 0..<numPolys {
                    let _ = try engine.open(testPolys[i], at: testPoints[i])
                }
                individualTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            individualTimes.sort()
            let indivMedian = individualTimes[2]

            // Time 1 multi-point batch open
            var batchTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.batchOpenMultiPoint(polynomials: testPolys, points: testPoints, gamma: testGamma)
                batchTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            batchTimes.sort()
            let batchMedian = batchTimes[2]

            let speedup = indivMedian / batchMedian
            print(String(format: "  N=%-2d deg=%d | %d individual: %6.1fms | 1 multi-pt batch: %6.1fms | speedup: %.1fx",
                        numPolys, polyDeg, numPolys, indivMedian, batchMedian, speedup))
        }

    } catch {
        print("  [FAIL] KZG batch error: \(error)")
    }

    print("\nKZG batch benchmark complete.")
}
