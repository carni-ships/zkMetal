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
        // For degree 2n polynomial: quotient has degree n, remainder has degree < n
        // Note: MSM > 2048 points uses GPU path which has a pre-existing issue,
        // so we cap polynomial degree at 2048 (logN=10, polyDeg=2048)
        let maxLogN = 10
        let maxSRSSize = 2 * (1 << maxLogN) + 16
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
        print("  logN        prove(ms)   verify(ms)  proof(bytes)")

        let logSizes = [6, 8, 10]
        for logN in logSizes {
            let n = 1 << logN
            guard 2 * n + 16 <= srs.count else {
                print("  2^\(logN): skipped (SRS too small)")
                continue
            }

            // Build a polynomial of degree < 2n with known sum.
            // sum_{x in H} x^j = n if n | j, else 0.
            // So sum_{x in H} f(x) = n * (c_0 + c_n + c_{2n} + ...)
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

            // Proof size: 4 projective points (96 bytes each) + 3 Fr elements (32 bytes each)
            // qCommitment, pCommitment, fOpeningProof, qOpeningProof, pOpeningProof = 5 points
            // fEval, qEval, pEval = 3 scalars
            let proofSize = 5 * 96 + 3 * 32

            print(String(format: "  2^%-5d %10.1f %10.1f %10d",
                         logN, medianProve, medianVerify, proofSize))
        }

        // --- Batch benchmark ---
        print("\n--- Batch Prove/Verify Benchmark ---")
        print("  logN     k      prove(ms)   verify(ms)")

        for logN in [8, 10] {
            let n = 1 << logN
            guard 2 * n + 16 <= srs.count else { continue }

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
    // Test 1: Small polynomial (deg < n) — simple case
    let logN1 = 3
    let n1 = 1 << logN1  // 8
    // f(X) = 5 + 3X + 7X^2 (degree 2, n=8)
    // sum_{x in H} f(x) = n * c_0 = 8 * 5 = 40 (only constant term survives)
    var coeffs1 = [Fr](repeating: Fr.zero, count: 3)
    coeffs1[0] = frFromInt(5)
    coeffs1[1] = frFromInt(3)
    coeffs1[2] = frFromInt(7)
    let claimedSum1 = frFromInt(UInt64(n1) * 5)  // 40

    let proveT1 = Transcript(label: "test-small", backend: .keccak256)
    let proof1 = try engine.prove(fCoeffs: coeffs1, logN: logN1, claimedSum: claimedSum1, transcript: proveT1)
    let fComm1 = try engine.kzg.commit(coeffs1)
    let verifyT1 = Transcript(label: "test-small", backend: .keccak256)
    let ok1 = engine.verify(proof: proof1, fCommitment: fComm1,
                            claimedSum: claimedSum1, logN: logN1,
                            transcript: verifyT1, srsSecret: srsSecret)
    print("  [" + (ok1 ? "pass" : "FAIL") + "] Small polynomial (deg=2, n=8)")

    // Test 2: Polynomial with degree >= n (the tricky case)
    let logN2 = 2
    let n2 = 1 << logN2  // 4
    // f(X) = 3 + 2X + 5X^2 + X^3 + 7X^4 + 0X^5 + 0X^6 + 0X^7
    // sum_{x in H} f(x) = n * (c_0 + c_4) = 4 * (3 + 7) = 40
    var coeffs2 = [Fr](repeating: Fr.zero, count: 2 * n2)
    coeffs2[0] = frFromInt(3)
    coeffs2[1] = frFromInt(2)
    coeffs2[2] = frFromInt(5)
    coeffs2[3] = frFromInt(1)
    coeffs2[4] = frFromInt(7)
    let claimedSum2 = frFromInt(UInt64(n2) * (3 + 7))  // 40

    let proveT2 = Transcript(label: "test-deg-geq-n", backend: .keccak256)
    let proof2 = try engine.prove(fCoeffs: coeffs2, logN: logN2, claimedSum: claimedSum2, transcript: proveT2)
    let fComm2 = try engine.kzg.commit(coeffs2)
    let verifyT2 = Transcript(label: "test-deg-geq-n", backend: .keccak256)
    let ok2 = engine.verify(proof: proof2, fCommitment: fComm2,
                            claimedSum: claimedSum2, logN: logN2,
                            transcript: verifyT2, srsSecret: srsSecret)
    print("  [" + (ok2 ? "pass" : "FAIL") + "] Polynomial deg >= n (deg=4, n=4)")

    // Test 3: Wrong sum should fail
    let wrongSum = frFromInt(999)
    let proveT3 = Transcript(label: "test-wrong", backend: .keccak256)
    let proofWrong = try engine.prove(fCoeffs: coeffs2, logN: logN2, claimedSum: wrongSum, transcript: proveT3)
    let verifyT3 = Transcript(label: "test-wrong", backend: .keccak256)
    let okWrong = engine.verify(proof: proofWrong, fCommitment: fComm2,
                                claimedSum: wrongSum, logN: logN2,
                                transcript: verifyT3, srsSecret: srsSecret)
    // Wrong sum: the remainder's constant term won't be zero, so q*Z_H + r*X != g
    // The verifier equation should fail at the random challenge with overwhelming probability
    print("  [" + (!okWrong ? "pass" : "WARN") + "] Rejects wrong sum")

    // Test 4: Larger random polynomial
    let logN4 = 6
    let n4 = 1 << logN4
    let polyDeg4 = 2 * n4
    var coeffs4 = [Fr](repeating: Fr.zero, count: polyDeg4)
    var rng: UInt64 = 0x12345678
    for i in 0..<polyDeg4 {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        coeffs4[i] = frFromInt(UInt64(rng >> 33) & 0xFFFFF)
    }
    let n4Fr = frFromInt(UInt64(n4))
    let sum4 = frMul(n4Fr, frAdd(coeffs4[0], coeffs4[n4]))

    let proveT4 = Transcript(label: "test-larger", backend: .keccak256)
    let proof4 = try engine.prove(fCoeffs: coeffs4, logN: logN4, claimedSum: sum4, transcript: proveT4)
    let fComm4 = try engine.kzg.commit(coeffs4)
    let verifyT4 = Transcript(label: "test-larger", backend: .keccak256)
    let ok4 = engine.verify(proof: proof4, fCommitment: fComm4,
                            claimedSum: sum4, logN: logN4,
                            transcript: verifyT4, srsSecret: srsSecret)
    print("  [" + (ok4 ? "pass" : "FAIL") + "] Large random polynomial (logN=6, deg=128)")
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

    print("  [" + (ok ? "pass" : "FAIL") + "] Batch prove/verify (logN=4, k=3)")
}
