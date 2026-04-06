// GPU KZG Engine — Benchmark and correctness tests
import zkMetal
import Foundation

public func runGPUKZGBench() {
    print("=== GPU KZG Engine Benchmark (BN254 G1) ===")

    do {
        // Setup: BN254 generator, test SRS
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let generator = PointAffine(x: gx, y: gy)

        let secretLimbs: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]
        let secret = frFromLimbs(secretLimbs)
        let srsSize = 4096
        let srs = KZGEngine.generateTestSRS(secret: secretLimbs, size: srsSize, generator: generator)

        let engine = try GPUKZGEngine(srs: srs)
        // Also create a KZGEngine for comparison
        let cpuEngine = try KZGEngine(srs: srs)

        // --- Correctness Tests ---
        print("\n--- Correctness ---")

        // Test 1: Commit matches KZGEngine
        let poly = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let gpuCommit = try engine.commit(poly)
        let cpuCommit = try cpuEngine.commit(poly)
        let gpuAff = batchToAffine([gpuCommit])[0]
        let cpuAff = batchToAffine([cpuCommit])[0]
        if fpToInt(gpuAff.x) == fpToInt(cpuAff.x) && fpToInt(gpuAff.y) == fpToInt(cpuAff.y) {
            print("  [pass] GPU commit matches KZGEngine commit")
        } else {
            print("  [FAIL] GPU commit does not match KZGEngine commit")
        }

        // Test 2: Open matches
        let z = frFromInt(5)
        let gpuOpen = try engine.open(poly, at: z)
        let cpuOpen = try cpuEngine.open(poly, at: z)
        let gpuEval = frToInt(gpuOpen.evaluation)[0]
        let cpuEval = frToInt(cpuOpen.evaluation)[0]
        if gpuEval == cpuEval && gpuEval == 86 {
            print("  [pass] GPU open: p(5) = 86 matches KZGEngine")
        } else {
            print("  [FAIL] GPU open: got \(gpuEval), KZGEngine got \(cpuEval), expected 86")
        }

        // Test 3: Batch open correctness
        let polys: [[Fr]] = [
            [frFromInt(1), frFromInt(2), frFromInt(3)],                // 1 + 2x + 3x^2
            [frFromInt(5), frFromInt(7)],                               // 5 + 7x
            [frFromInt(10), frFromInt(0), frFromInt(1), frFromInt(4)],  // 10 + x^2 + 4x^3
            [frFromInt(3), frFromInt(3), frFromInt(3), frFromInt(3)],   // 3 + 3x + 3x^2 + 3x^3
        ]

        let gamma = frFromInt(17)
        let batch = try engine.batchOpen(polys: polys, point: z, gamma: gamma)

        // Expected: p0(5)=86, p1(5)=40, p2(5)=535, p3(5)=468
        let expectedEvals: [UInt64] = [86, 40, 535, 468]
        var evalPass = true
        for i in 0..<4 {
            let got = frToInt(batch.evaluations[i])[0]
            if got != expectedEvals[i] {
                print("  [FAIL] Batch eval p\(i)(5) = \(got), expected \(expectedEvals[i])")
                evalPass = false
            }
        }
        if evalPass {
            print("  [pass] Batch open: all 4 evaluations correct")
        }

        // Test 4: Batch verify with known secret
        let verifyOk = engine.batchVerifyProof(batch, srsSecret: secret)
        if verifyOk {
            print("  [pass] Batch verify accepts valid proof")
        } else {
            print("  [FAIL] Batch verify rejected valid proof")
        }

        // Test 5: Reject tampered evaluation
        var tamperedProof = batch
        var tamperedEvals = batch.evaluations
        tamperedEvals[0] = frFromInt(999)
        tamperedProof = GPUBatchProof(
            commitments: batch.commitments, evaluations: tamperedEvals,
            witness: batch.witness, point: batch.point, gamma: batch.gamma)
        let tamperedOk = engine.batchVerify(
            commitments: tamperedProof.commitments,
            point: tamperedProof.point,
            values: tamperedProof.evaluations,
            proof: tamperedProof,
            srsSecret: secret)
        if !tamperedOk {
            print("  [pass] Batch verify rejects tampered evaluation")
        } else {
            print("  [FAIL] Batch verify accepted tampered evaluation")
        }

        // Test 6: Multi-point batch open
        let multiPoints = [frFromInt(5), frFromInt(7), frFromInt(3), frFromInt(11)]
        let multiProof = try engine.batchOpenMultiPoint(polys: polys, points: multiPoints, gamma: gamma)
        let expectedMultiEvals: [UInt64] = [86, 54, 127, 4392]
        var multiPass = true
        for i in 0..<4 {
            let got = frToInt(multiProof.evaluations[i])[0]
            if got != expectedMultiEvals[i] {
                print("  [FAIL] Multi-point eval p\(i)(z\(i)) = \(got), expected \(expectedMultiEvals[i])")
                multiPass = false
            }
        }
        if multiPass {
            print("  [pass] Multi-point batch open: all 4 evaluations correct")
        }

        // Test 7: Multi-point verify with known secret
        let multiVerifyOk = engine.batchVerifyMultiPoint(
            commitments: multiProof.commitments,
            points: multiPoints,
            values: multiProof.evaluations,
            witness: multiProof.witness,
            gamma: gamma,
            srsSecret: secret)
        if multiVerifyOk {
            print("  [pass] Multi-point batch verify accepts valid proof")
        } else {
            print("  [FAIL] Multi-point batch verify rejected valid proof")
        }

        // --- Performance: GPU KZG vs standard KZGEngine ---
        print("\n--- Commit Performance: GPU KZG vs KZGEngine ---")

        for logN in [8, 10, 12] {
            let n = 1 << logN
            guard n <= srsSize else { continue }

            var coeffs = [Fr](repeating: Fr.zero, count: n)
            var rng: UInt64 = 0xDEAD_BEEF
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                coeffs[i] = frFromInt(rng >> 32)
            }

            // Warmup
            let _ = try engine.commit(coeffs)
            let _ = try cpuEngine.commit(coeffs)

            // GPU KZG
            var gpuTimes = [Double]()
            for _ in 0..<5 {
                let t = CFAbsoluteTimeGetCurrent()
                let _ = try engine.commit(coeffs)
                gpuTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
            }
            gpuTimes.sort()

            // Standard KZGEngine
            var stdTimes = [Double]()
            for _ in 0..<5 {
                let t = CFAbsoluteTimeGetCurrent()
                let _ = try cpuEngine.commit(coeffs)
                stdTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
            }
            stdTimes.sort()

            print(String(format: "  Commit 2^%-2d | GPU KZG: %6.2fms | KZGEngine: %6.2fms | ratio: %.2fx",
                        logN, gpuTimes[2], stdTimes[2], stdTimes[2] / gpuTimes[2]))
        }

        // --- Batch open performance ---
        print("\n--- Batch Open Performance ---")

        for numPolys in [4, 8, 16] {
            let polyDeg = min(512, srsSize)
            var rng: UInt64 = 0xCAFE_BABE_0000 &+ UInt64(numPolys)
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
            let _ = try engine.batchOpen(polys: testPolys, point: testZ, gamma: testGamma)

            // N individual opens
            var indivTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                for p in testPolys {
                    let _ = try engine.open(p, at: testZ)
                }
                indivTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            indivTimes.sort()

            // 1 batch open
            var batchTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.batchOpen(polys: testPolys, point: testZ, gamma: testGamma)
                batchTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            batchTimes.sort()

            let speedup = indivTimes[2] / batchTimes[2]
            print(String(format: "  N=%-2d deg=%d | %d individual: %6.1fms | 1 batch: %6.1fms | speedup: %.1fx",
                        numPolys, polyDeg, numPolys, indivTimes[2], batchTimes[2], speedup))
        }

    } catch {
        print("  [FAIL] GPU KZG error: \(error)")
    }

    print("\nGPU KZG benchmark complete.")
}
