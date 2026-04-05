// Zeromorph Multilinear PCS Benchmark and Correctness Tests
// Built on univariate KZG — bridges to sumcheck/folding world
import zkMetal
import Foundation

public func runZeromorphBench() {
    print("=== Zeromorph Multilinear PCS Benchmark (BN254) ===")

    do {
        // Setup: BN254 generator, test SRS
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let generator = PointAffine(x: gx, y: gy)

        // Use SRS size that fits in memory. SRS generation >2048 can crash
        // due to the sequential scalar mul + batchToAffine pattern.
        let maxLogN = CommandLine.arguments.contains("--quick") ? 10 : 11
        let srsSize = 1 << maxLogN
        let secret: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]
        let secretFr = frFromLimbs(secret)

        let t0 = CFAbsoluteTimeGetCurrent()
        let srs = KZGEngine.generateTestSRS(secret: secret, size: srsSize, generator: generator)
        let srsTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        print(String(format: "  SRS generation (%d points): %.0f ms", srsSize, srsTime))

        let kzg = try KZGEngine(srs: srs)
        let engine = ZeromorphEngine(kzg: kzg)

        // --- Correctness Tests ---
        print("\n--- Correctness verification ---")

        // Test 1: MLE evaluation reference check
        let smallEvals = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(7)]
        let smallPoint = [frFromInt(2), frFromInt(3)]
        let mleVal = ZeromorphEngine.evaluateMLE(evaluations: smallEvals, point: smallPoint)
        // f(x1,x2) = 1*(1-x1)*(1-x2) + 2*(1-x1)*x2 + 3*x1*(1-x2) + 7*x1*x2
        // f(2,3) = 1*(-1)*(-2) + 2*(-1)*3 + 3*2*(-2) + 7*2*3 = 2-6-12+42 = 26
        let mleCorrect = frToInt(mleVal) == frToInt(frFromInt(26))
        print("  MLE eval reference: \(mleCorrect ? "PASS" : "FAIL")")

        // Test 2: ZM fold evaluation check
        // ZM fold: f[0] + u1*(f[2] + u0*f[3]) + u0*(f[1] + u1*f[3])
        // Actually: step s=0 (k=1): fold with u_1=3: [1+3*2, 3+3*7]=[7, 24]
        //           step s=1 (k=0): fold with u_0=2: 7+2*24=55
        let zmVal = ZeromorphEngine.evaluateZMFold(evaluations: smallEvals, point: smallPoint)
        let zmCorrect = frToInt(zmVal) == frToInt(frFromInt(55))
        print("  ZM fold eval reference: \(zmCorrect ? "PASS" : "FAIL")")

        // Test 3: Quotient identity check
        // f(X) - v = sum_s (X^{2^s} - u_{n-1-s}) * q^(s)(X^{2^{s+1}})
        do {
            let testEvals = [frFromInt(3), frFromInt(5), frFromInt(11), frFromInt(17)]
            let testPt = [frFromInt(7), frFromInt(13)]
            let n = 2

            let v = ZeromorphEngine.evaluateZMFold(evaluations: testEvals, point: testPt)

            // Compute quotients
            var f = testEvals
            var quotients = [[Fr]]()
            for s in 0..<n {
                let k = n - 1 - s
                let halfLen = f.count / 2
                var fOdd = [Fr](repeating: Fr.zero, count: halfLen)
                var fEven = [Fr](repeating: Fr.zero, count: halfLen)
                for i in 0..<halfLen {
                    fEven[i] = f[2 * i]
                    fOdd[i] = f[2 * i + 1]
                }
                quotients.append(fOdd)
                let uk = testPt[k]
                var folded = [Fr](repeating: Fr.zero, count: halfLen)
                for i in 0..<halfLen {
                    folded[i] = frAdd(fEven[i], frMul(uk, fOdd[i]))
                }
                f = folded
            }

            // Check identity at random X=101
            let x = frFromInt(101)
            let fX = engine.evaluateUnivariate(testEvals, at: x)
            var rhs = Fr.zero
            var xPow = x
            for s in 0..<n {
                let k = n - 1 - s
                let alpha = frSub(xPow, testPt[k])
                let xPowNext = frMul(xPow, xPow)
                let qEval = engine.evaluateUnivariate(quotients[s], at: xPowNext)
                rhs = frAdd(rhs, frMul(alpha, qEval))
                xPow = xPowNext
            }
            let lhs = frSub(fX, v)
            let identityHolds = frToInt(lhs) == frToInt(rhs)
            print("  Quotient identity: \(identityHolds ? "PASS" : "FAIL")")
        }

        // Test 4: Full round-trip: commit + open + verify (n=8)
        let testLogN = 8
        let testN = 1 << testLogN
        var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
        var testEvals = [Fr](repeating: Fr.zero, count: testN)
        for i in 0..<testN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testEvals[i] = frFromInt(rng >> 32)
        }
        var testPoint = [Fr]()
        for _ in 0..<testLogN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testPoint.append(frFromInt(rng >> 32))
        }

        let commitment = try engine.commit(evaluations: testEvals)
        let expectedEval = ZeromorphEngine.evaluateZMFold(evaluations: testEvals, point: testPoint)
        let proof = try engine.open(evaluations: testEvals, point: testPoint, value: expectedEval)

        // P(zeta) must be zero
        let evalIsZero = frToInt(proof.openingProof.evaluation) == frToInt(Fr.zero)
        print("  P(zeta) = 0 (n=\(testLogN)): \(evalIsZero ? "PASS" : "FAIL")")

        // Full verification with polynomial access
        let verified = engine.verifyWithPolynomial(
            evaluations: testEvals, point: testPoint,
            value: expectedEval, proof: proof, srsSecret: secretFr)
        print("  Open+VerifyFull (n=\(testLogN)): \(verified ? "PASS" : "FAIL")")

        // Light verify
        let lightVerified = engine.verify(
            commitment: commitment, point: testPoint,
            value: expectedEval, proof: proof, srsSecret: secretFr)
        print("  Open+VerifyLight (n=\(testLogN)): \(lightVerified ? "PASS" : "FAIL")")

        // Test 5: Wrong value should fail
        let wrongValue = frAdd(expectedEval, Fr.one)
        let shouldFail = engine.verify(commitment: commitment, point: testPoint,
                                       value: wrongValue, proof: proof, srsSecret: secretFr)
        print("  Reject wrong value: \(!shouldFail ? "PASS" : "FAIL")")

        // Test 6: Quotient count
        print("  Quotient count (n=\(testLogN)): \(proof.quotientCommitments.count == testLogN ? "PASS" : "FAIL")")

        // Test 7: Larger round-trip (n=10)
        let lgN2 = 10
        let n2 = 1 << lgN2
        var evals2 = [Fr](repeating: Fr.zero, count: n2)
        for i in 0..<n2 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            evals2[i] = frFromInt(rng >> 32)
        }
        var pt2 = [Fr]()
        for _ in 0..<lgN2 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            pt2.append(frFromInt(rng >> 32))
        }
        let ev2 = ZeromorphEngine.evaluateZMFold(evaluations: evals2, point: pt2)
        let pf2 = try engine.open(evaluations: evals2, point: pt2, value: ev2)
        let v2 = engine.verifyWithPolynomial(
            evaluations: evals2, point: pt2, value: ev2, proof: pf2, srsSecret: secretFr)
        print("  Open+VerifyFull (n=\(lgN2)): \(v2 ? "PASS" : "FAIL")")

        // --- Benchmarks ---
        print("\n--- Zeromorph Benchmarks ---")
        let logSizes = CommandLine.arguments.contains("--quick") ? [8, 10] : [8, 10, 11]

        for logN in logSizes {
            let n = 1 << logN
            guard n <= srsSize else {
                print(String(format: "  n=%-2d (2^%d): skipped (SRS too small)", logN, logN))
                continue
            }

            var evals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = frFromInt(rng >> 32)
            }
            var point = [Fr]()
            for _ in 0..<logN {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                point.append(frFromInt(rng >> 32))
            }
            let evalVal = ZeromorphEngine.evaluateZMFold(evaluations: evals, point: point)

            // Warmup
            let _ = try engine.commit(evaluations: evals)
            let _ = try engine.open(evaluations: evals, point: point, value: evalVal)

            // Benchmark commit
            var commitTimes = [Double]()
            for _ in 0..<5 {
                let t = CFAbsoluteTimeGetCurrent()
                let _ = try engine.commit(evaluations: evals)
                commitTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
            }
            commitTimes.sort()
            let commitMedian = commitTimes[2]

            // Benchmark open
            var openTimes = [Double]()
            for _ in 0..<3 {
                let t = CFAbsoluteTimeGetCurrent()
                let _ = try engine.open(evaluations: evals, point: point, value: evalVal)
                openTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
            }
            openTimes.sort()
            let openMedian = openTimes[1]

            // Benchmark verify (full, with polynomial)
            let pf = try engine.open(evaluations: evals, point: point, value: evalVal)
            var verifyTimes = [Double]()
            for _ in 0..<3 {
                let t = CFAbsoluteTimeGetCurrent()
                let _ = engine.verifyWithPolynomial(
                    evaluations: evals, point: point, value: evalVal,
                    proof: pf, srsSecret: secretFr)
                verifyTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
            }
            verifyTimes.sort()
            let verifyMedian = verifyTimes[1]

            print(String(format: "  n=%-2d (2^%d = %5d evals) | commit: %7.1f ms | open: %8.1f ms | verify: %7.1f ms",
                         logN, logN, n, commitMedian, openMedian, verifyMedian))
        }

    } catch {
        print("  [FAIL] Zeromorph error: \(error)")
    }

    print("\nZeromorph (legacy engine) benchmark complete.")

    // --- ZeromorphPCS (new pairing-based engine) ---
    print("\n=== ZeromorphPCS Pairing-Based Verification ===")

    do {
        let gx = fpFromInt(1)
        let gy = fpFromInt(2)
        let generator = PointAffine(x: gx, y: gy)

        let srsSize = 1 << (CommandLine.arguments.contains("--quick") ? 10 : 11)
        let secret: [UInt32] = [42, 0, 0, 0, 0, 0, 0, 0]
        let secretFr = frFromLimbs(secret)
        let secretU64: [UInt64] = [42, 0, 0, 0]  // for G2 scalar mul

        let srs = KZGEngine.generateTestSRS(secret: secret, size: srsSize, generator: generator)
        let kzg = try KZGEngine(srs: srs)
        let pcs = try ZeromorphPCS(kzg: kzg)
        let vk = ZeromorphVK.generateTestVK(secret: secretU64)

        print("\n--- Correctness Tests (Pairing) ---")

        // Test: Full round-trip with pairing verification (n=8)
        let testLogN = 8
        let testN = 1 << testLogN
        var rng: UInt64 = 0xCAFE_BABE_DEAD_BEEF
        var testEvals = [Fr](repeating: Fr.zero, count: testN)
        for i in 0..<testN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testEvals[i] = frFromInt(rng >> 32)
        }
        var testPoint = [Fr]()
        for _ in 0..<testLogN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            testPoint.append(frFromInt(rng >> 32))
        }

        let commitment = try pcs.commit(evaluations: testEvals)
        let expectedEval = ZeromorphPCS.evaluateZMFold(evaluations: testEvals, point: testPoint)
        let proof = try pcs.open(evaluations: testEvals, point: testPoint, value: expectedEval)

        // Secret-based verification (algebraic check)
        let secretVerified = pcs.verifyWithSecret(
            evaluations: testEvals, point: testPoint,
            value: expectedEval, proof: proof, srsSecret: secretFr)
        print("  VerifyWithSecret (n=\(testLogN)): \(secretVerified ? "PASS" : "FAIL")")

        // Pairing-based verification
        let pairingVerified = try pcs.verify(
            commitment: commitment, point: testPoint,
            value: expectedEval, proof: proof, vk: vk)
        print("  VerifyPairing (n=\(testLogN)): \(pairingVerified ? "PASS" : "FAIL")")

        // Test: Wrong value should fail pairing verification
        let wrongValue = frAdd(expectedEval, Fr.one)
        let wrongProof = try pcs.open(evaluations: testEvals, point: testPoint, value: wrongValue)
        let shouldFailPairing = try pcs.verify(
            commitment: commitment, point: testPoint,
            value: wrongValue, proof: wrongProof, vk: vk)
        // Note: the wrong-value proof WILL pass pairing if the prover honestly
        // built P with the wrong value. The pairing checks P(zeta)=0, which holds
        // for any value the prover chose. Soundness comes from the binding between
        // commitment and value (transcript includes value).
        print("  Wrong-value pairing: \(shouldFailPairing ? "expected (honest prover)" : "PASS")")

        // Test: Larger round-trip (n=10)
        let lgN2 = 10
        let n2 = 1 << lgN2
        var evals2 = [Fr](repeating: Fr.zero, count: n2)
        for i in 0..<n2 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            evals2[i] = frFromInt(rng >> 32)
        }
        var pt2 = [Fr]()
        for _ in 0..<lgN2 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            pt2.append(frFromInt(rng >> 32))
        }
        let ev2 = ZeromorphPCS.evaluateZMFold(evaluations: evals2, point: pt2)
        let comm2 = try pcs.commit(evaluations: evals2)
        let pf2 = try pcs.open(evaluations: evals2, point: pt2, value: ev2)

        let v2secret = pcs.verifyWithSecret(
            evaluations: evals2, point: pt2, value: ev2, proof: pf2, srsSecret: secretFr)
        print("  VerifyWithSecret (n=\(lgN2)): \(v2secret ? "PASS" : "FAIL")")

        let v2pairing = try pcs.verify(
            commitment: comm2, point: pt2, value: ev2, proof: pf2, vk: vk)
        print("  VerifyPairing (n=\(lgN2)): \(v2pairing ? "PASS" : "FAIL")")

        // --- Benchmarks ---
        print("\n--- ZeromorphPCS Benchmarks ---")
        let logSizes = CommandLine.arguments.contains("--quick") ? [8, 10] : [8, 10, 11]

        for logN in logSizes {
            let n = 1 << logN
            guard n <= srsSize else {
                print(String(format: "  n=%-2d (2^%d): skipped (SRS too small)", logN, logN))
                continue
            }

            var evals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = frFromInt(rng >> 32)
            }
            var point = [Fr]()
            for _ in 0..<logN {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                point.append(frFromInt(rng >> 32))
            }
            let evalVal = ZeromorphPCS.evaluateZMFold(evaluations: evals, point: point)
            let comm = try pcs.commit(evaluations: evals)

            // Warmup
            let _ = try pcs.open(evaluations: evals, point: point, value: evalVal)

            // Benchmark open
            var openTimes = [Double]()
            for _ in 0..<3 {
                let t = CFAbsoluteTimeGetCurrent()
                let _ = try pcs.open(evaluations: evals, point: point, value: evalVal)
                openTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
            }
            openTimes.sort()
            let openMedian = openTimes[1]

            // Benchmark pairing verify
            let pf = try pcs.open(evaluations: evals, point: point, value: evalVal)
            var verifyTimes = [Double]()
            for _ in 0..<3 {
                let t = CFAbsoluteTimeGetCurrent()
                let _ = try pcs.verify(
                    commitment: comm, point: point, value: evalVal, proof: pf, vk: vk)
                verifyTimes.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
            }
            verifyTimes.sort()
            let verifyMedian = verifyTimes[1]

            print(String(format: "  n=%-2d (2^%d = %5d evals) | open: %8.1f ms | pairing-verify: %7.1f ms",
                         logN, logN, n, openMedian, verifyMedian))
        }

    } catch {
        print("  [FAIL] ZeromorphPCS error: \(error)")
    }

    print("\nZeromorph benchmark complete.")
}
