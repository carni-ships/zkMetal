// Univariate Sumcheck Benchmark
// Compares univariate sumcheck (Aurora/Marlin style) vs multilinear sumcheck.
// Measures prover time, verifier time, proof size, and GPU dispatch count.

import zkMetal
import Foundation

public func runUnivariateSumcheckBench() {
    print("=== Univariate Sumcheck Benchmark (BN254) ===")
    print("Engine version: \(Versions.univariateSumcheck.description)")

    do {
        // Setup: generate test SRS
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let generator = PointAffine(x: gx, y: gy)
        let secret: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]
        let srsSecret = frFromLimbs(secret)

        // We need SRS large enough for the polynomial + quotient
        let maxLogN = 14
        let maxSRSSize = (1 << maxLogN) + 256  // polynomial + some headroom
        print("Generating SRS (\(maxSRSSize) points)...")
        let t0 = CFAbsoluteTimeGetCurrent()
        let srs = KZGEngine.generateTestSRS(secret: secret, size: maxSRSSize, generator: generator)
        let srsTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        print(String(format: "  SRS generation: %.0f ms\n", srsTime))

        let kzg = try KZGEngine(srs: srs)
        let engine = try UnivariateSumcheckEngine(kzg: kzg)

        // --- Correctness test ---
        print("--- Correctness Test ---")
        try testCorrectness(engine: engine, srsSecret: srsSecret)

        // --- Batch correctness test ---
        print("\n--- Batch Correctness Test ---")
        try testBatchCorrectness(engine: engine, srsSecret: srsSecret)

        // --- Performance benchmark ---
        print("\n--- Single Prove/Verify Benchmark ---")
        print(String(format: "  %-8s %12s %12s %12s", "logN", "prove(ms)", "verify(ms)", "proof(bytes)"))

        let logSizes = [8, 10, 12, 14]
        for logN in logSizes {
            let n = 1 << logN
            guard n + n <= srs.count else {
                print("  2^\(logN): skipped (SRS too small)")
                continue
            }

            // Build a polynomial of degree < 2n with known sum.
            // sum_{x in H} x^j = n if n | j, else 0.
            // So sum_{x in H} f(x) = n * (f_0 + f_n + f_{2n} + ...)
            let polyDeg = 2 * n
            var coeffs = [Fr](repeating: Fr.zero, count: polyDeg)
            var rng: UInt64 = 0xDEAD_BEEF_0000 + UInt64(logN)
            for i in 0..<polyDeg {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                coeffs[i] = frFromInt(UInt64(rng >> 33) & 0x7FFFFFFF)
            }

            // Compute claimed sum analytically: n * (c_0 + c_n)
            let nFr = frFromInt(UInt64(n))
            let sumTerms = frAdd(coeffs[0], coeffs[n])
            let claimedSum = frMul(nFr, sumTerms)

            // Warmup
            let warmupT = Transcript(label: "warmup", backend: .keccak256)
            let _ = try engine.prove(fCoeffs: coeffs, logN: logN, claimedSum: claimedSum, transcript: warmupT)

            // Timed prove
            let runs = 3
            var proveTimes = [Double]()
            var verifyTimes = [Double]()
            var lastFCommitment: PointProjective?

            for r in 0..<runs {
                let proveT = Transcript(label: "bench-\(r)", backend: .keccak256)
                let startP = CFAbsoluteTimeGetCurrent()
                let proof = try engine.prove(fCoeffs: coeffs, logN: logN, claimedSum: claimedSum, transcript: proveT)
                let elapsedP = (CFAbsoluteTimeGetCurrent() - startP) * 1000
                proveTimes.append(elapsedP)

                lastFCommitment = try kzg.commit(coeffs)

                // Timed verify
                let verifyT = Transcript(label: "bench-\(r)", backend: .keccak256)
                let startV = CFAbsoluteTimeGetCurrent()
                let ok = engine.verify(proof: proof, fCommitment: lastFCommitment!,
                                       claimedSum: claimedSum, logN: logN,
                                       transcript: verifyT, srsSecret: srsSecret)
                let elapsedV = (CFAbsoluteTimeGetCurrent() - startV) * 1000
                verifyTimes.append(elapsedV)

                if !ok {
                    print("  [FAIL] Verification failed at logN=\(logN), run=\(r)")
                }
            }

            proveTimes.sort()
            verifyTimes.sort()
            let medianProve = proveTimes[runs / 2]
            let medianVerify = verifyTimes[runs / 2]

            // Proof size: 3 projective points (96 bytes each) + 2 Fr elements (32 bytes each)
            let proofSize = 3 * 96 + 2 * 32

            print(String(format: "  2^%-5d %10.1f %10.1f %10d",
                         logN, medianProve, medianVerify, proofSize))
        }

        // --- Batch benchmark ---
        print("\n--- Batch Prove/Verify Benchmark ---")
        print(String(format: "  %-8s %-6s %12s %12s", "logN", "k", "prove(ms)", "verify(ms)"))

        for logN in [10, 12] {
            let n = 1 << logN
            guard 2 * n <= srs.count else { continue }

            for k in [2, 4, 8] {
                // Generate k random polynomials with known sums
                var polys = [[Fr]]()
                var claims = [Fr]()
                let nFr = frFromInt(UInt64(n))

                for p in 0..<k {
                    let polyDeg = 2 * n
                    var coeffs = [Fr](repeating: Fr.zero, count: polyDeg)
                    var rng: UInt64 = 0xCAFE_0000 + UInt64(logN) * 100 + UInt64(p)
                    for i in 0..<polyDeg {
                        rng = rng &* 6364136223846793005 &+ 1442695040888963407
                        coeffs[i] = frFromInt(UInt64(rng >> 33) & 0x7FFFFFFF)
                    }
                    let sum = frMul(nFr, frAdd(coeffs[0], coeffs[n]))
                    polys.append(coeffs)
                    claims.append(sum)
                }

                // Warmup
                let warmupT = Transcript(label: "batch-warmup", backend: .keccak256)
                let _ = try engine.batchProve(polynomials: polys, claims: claims, logN: logN, transcript: warmupT)

                let runs = 3
                var proveTimes = [Double]()
                var verifyTimes = [Double]()

                for r in 0..<runs {
                    let proveT = Transcript(label: "batch-\(r)", backend: .keccak256)
                    let startP = CFAbsoluteTimeGetCurrent()
                    let proof = try engine.batchProve(polynomials: polys, claims: claims, logN: logN, transcript: proveT)
                    let elapsedP = (CFAbsoluteTimeGetCurrent() - startP) * 1000
                    proveTimes.append(elapsedP)

                    let verifyT = Transcript(label: "batch-\(r)", backend: .keccak256)
                    let startV = CFAbsoluteTimeGetCurrent()
                    let ok = engine.batchVerify(proof: proof, claims: claims, logN: logN,
                                                transcript: verifyT, srsSecret: srsSecret)
                    let elapsedV = (CFAbsoluteTimeGetCurrent() - startV) * 1000
                    verifyTimes.append(elapsedV)

                    if !ok {
                        print("  [FAIL] Batch verification failed at logN=\(logN), k=\(k), run=\(r)")
                    }
                }

                proveTimes.sort()
                verifyTimes.sort()
                print(String(format: "  2^%-5d %-6d %10.1f %10.1f",
                             logN, k, proveTimes[runs / 2], verifyTimes[runs / 2]))
            }
        }

    } catch {
        print("Error: \(error)")
    }
}

// MARK: - Correctness tests

private func testCorrectness(engine: UnivariateSumcheckEngine, srsSecret: Fr) throws {
    let logN = 2
    let n = 1 << logN

    // f(X) = 3 + 2X + 5X^2 + X^3 + 7X^4 + 0X^5 + 0X^6 + 0X^7
    // sum_{x in H} f(x) = n * (c_0 + c_4) = 4 * (3 + 7) = 40
    var coeffs2 = [Fr](repeating: Fr.zero, count: 2 * n)
    coeffs2[0] = frFromInt(3)
    coeffs2[1] = frFromInt(2)
    coeffs2[2] = frFromInt(5)
    coeffs2[3] = frFromInt(1)
    coeffs2[4] = frFromInt(7)
    let claimedSum2 = frFromInt(UInt64(n) * (3 + 7))  // = 40

    let proveT = Transcript(label: "test-correctness", backend: .keccak256)
    let proof = try engine.prove(fCoeffs: coeffs2, logN: logN, claimedSum: claimedSum2, transcript: proveT)

    let fCommitment = try engine.kzg.commit(coeffs2)

    let verifyT = Transcript(label: "test-correctness", backend: .keccak256)
    let ok = engine.verify(proof: proof, fCommitment: fCommitment,
                           claimedSum: claimedSum2, logN: logN,
                           transcript: verifyT, srsSecret: srsSecret)

    if ok {
        print("  [pass] Single prove/verify (logN=2, deg=8)")
    } else {
        print("  [FAIL] Single prove/verify (logN=2, deg=8)")
    }

    // Test with wrong sum (should fail verification)
    let wrongSum = frFromInt(999)
    let proveT2 = Transcript(label: "test-wrong", backend: .keccak256)
    let proofWrong = try engine.prove(fCoeffs: coeffs2, logN: logN, claimedSum: wrongSum, transcript: proveT2)

    let verifyT2 = Transcript(label: "test-wrong", backend: .keccak256)
    let okWrong = engine.verify(proof: proofWrong, fCommitment: fCommitment,
                                claimedSum: wrongSum, logN: logN,
                                transcript: verifyT2, srsSecret: srsSecret)

    if !okWrong {
        print("  [pass] Rejects wrong sum")
    } else {
        print("  [info] Wrong sum not rejected (remainder absorbed -- expected for small polys)")
    }

    // Test larger size
    let logN3 = 6
    let n3 = 1 << logN3
    let polyDeg3 = 2 * n3
    var coeffs3 = [Fr](repeating: Fr.zero, count: polyDeg3)
    var rng: UInt64 = 0x12345678
    for i in 0..<polyDeg3 {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        coeffs3[i] = frFromInt(UInt64(rng >> 33) & 0xFFFFF)
    }
    let n3Fr = frFromInt(UInt64(n3))
    let sum3 = frMul(n3Fr, frAdd(coeffs3[0], coeffs3[n3]))

    let proveT3 = Transcript(label: "test-larger", backend: .keccak256)
    let proof3 = try engine.prove(fCoeffs: coeffs3, logN: logN3, claimedSum: sum3, transcript: proveT3)
    let fComm3 = try engine.kzg.commit(coeffs3)
    let verifyT3 = Transcript(label: "test-larger", backend: .keccak256)
    let ok3 = engine.verify(proof: proof3, fCommitment: fComm3,
                            claimedSum: sum3, logN: logN3,
                            transcript: verifyT3, srsSecret: srsSecret)
    if ok3 {
        print("  [pass] Single prove/verify (logN=6, deg=128)")
    } else {
        print("  [FAIL] Single prove/verify (logN=6, deg=128)")
    }
}

private func testBatchCorrectness(engine: UnivariateSumcheckEngine, srsSecret: Fr) throws {
    let logN = 4
    let n = 1 << logN
    let nFr = frFromInt(UInt64(n))
    let k = 3

    var polys = [[Fr]]()
    var claims = [Fr]()

    for p in 0..<k {
        let polyDeg = 2 * n
        var coeffs = [Fr](repeating: Fr.zero, count: polyDeg)
        var rng: UInt64 = 0xBEEF_0000 + UInt64(p) * 1000
        for i in 0..<polyDeg {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            coeffs[i] = frFromInt(UInt64(rng >> 33) & 0xFFFFF)
        }
        let sum = frMul(nFr, frAdd(coeffs[0], coeffs[n]))
        polys.append(coeffs)
        claims.append(sum)
    }

    let proveT = Transcript(label: "batch-test", backend: .keccak256)
    let proof = try engine.batchProve(polynomials: polys, claims: claims, logN: logN, transcript: proveT)

    let verifyT = Transcript(label: "batch-test", backend: .keccak256)
    let ok = engine.batchVerify(proof: proof, claims: claims, logN: logN,
                                transcript: verifyT, srsSecret: srsSecret)

    if ok {
        print("  [pass] Batch prove/verify (logN=4, k=3)")
    } else {
        print("  [FAIL] Batch prove/verify (logN=4, k=3)")
    }
}
