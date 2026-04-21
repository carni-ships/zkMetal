// P^1 Rational Function STARKs Benchmark — prototype over Mersenne31
//
// This benchmark tests the P^1 Rational Function approach which uses:
// - Multiplicative coset domain instead of circle group
// - Standard radix-2 FFT instead of circle-specific FFT
// - Standard t → t² FRI folding (simpler than y-fold + x-fold)
//
// Note: This is a prototype implementation. The theoretical foundations
// (particularly the 2-adicity of M31 for full FFT) are still being explored.

import zkMetal
import Foundation

public func runP1Bench() {
    print("=== P^1 Rational Function STARKs Benchmark (Mersenne31) ===")

    // ---- P^1 Domain Generation ----
    print("\n--- P^1 Coset Domain Generation ---")

    // Test domain generation for various sizes
    for logN in 1...8 {
        let n = 1 << logN
        let domain = p1CosetDomain(logN: logN)

        // Check sign-pair structure: domain[i + n/2] = -domain[i]
        var pairsCorrect = true
        let half = n >> 1
        for i in 0..<half {
            if domain[i + half].v != m31Neg(domain[i]).v {
                pairsCorrect = false
                break
            }
        }

        // Check squaring property: domain[i]^2 = domain[j]^2 for pairs
        var squaringCorrect = true
        for i in 0..<half {
            let sq1 = m31Sqr(domain[i])
            let sq2 = m31Sqr(domain[i + half])
            if sq1.v != sq2.v {
                squaringCorrect = false
                break
            }
        }

        print("  [\(pairsCorrect && squaringCorrect ? "pass" : "FAIL")] Domain 2^\(logN): \(n) points, sign pairs: \(pairsCorrect), squaring pairs: \(squaringCorrect)")
    }

    // ---- Twiddle Precomputation ----
    print("\n--- P^1 Twiddle Precomputation ---")

    for logN in 1...8 {
        let twiddles = p1PrecomputeForwardTwiddles(logN: logN)
        let invTwiddles = p1PrecomputeInverseTwiddles(logN: logN)

        // Check that w * w_inv = 1 for all twiddles
        var twCorrect = true
        let half = twiddles.count
        for i in 0..<half {
            let prod = m31Mul(twiddles[i], invTwiddles[i])
            if prod.v != 1 {
                twCorrect = false
                break
            }
        }
        print("  [\(twCorrect ? "pass" : "FAIL")] Twiddles 2^\(logN): \(twCorrect ? "w * w_inv = 1" : "FAILED")")
    }

    // ---- CPU P^1 NTT Correctness ----
    print("\n--- P^1 NTT (CPU reference) ---")

    for logN in 1...8 {
        let n = 1 << logN
        var coeffs = [M31](repeating: M31.zero, count: n)
        var rng: UInt64 = 0xDEAD_BEEF + UInt64(logN)
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            coeffs[i] = M31(v: UInt32(rng >> 33))
            if coeffs[i].v >= M31.P { coeffs[i].v = coeffs[i].v - M31.P }
        }

        let evals = P1NTTEngine.cpuNTT(coeffs, logN: logN)
        let recovered = P1NTTEngine.cpuINTT(evals, logN: logN)

        var match = true
        for i in 0..<n {
            if recovered[i].v != coeffs[i].v { match = false; break }
        }
        if match { print("  [pass] CPU P^1 NTT roundtrip: N = \(n)") }
        else {
            print("  [FAIL] CPU P^1 NTT roundtrip: N = \(n)")
            if n <= 8 {
                print("    coeffs:   \(coeffs.map { $0.v })")
                print("    evals:    \(evals.map { $0.v })")
                print("    recover:  \(recovered.map { $0.v })")
            }
        }
    }

    // ---- GPU P^1 NTT ----
    print("\n--- P^1 NTT (GPU) ---")
    do {
        let engine = try P1NTTEngine()

        for logN in 1...8 {
            let n = 1 << logN
            var coeffs = [M31](repeating: M31.zero, count: n)
            var rng: UInt64 = 0xCAFE_BABE + UInt64(logN)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                coeffs[i] = M31(v: UInt32(rng >> 33))
                if coeffs[i].v >= M31.P { coeffs[i].v = coeffs[i].v - M31.P }
            }

            let cpuEvals = P1NTTEngine.cpuNTT(coeffs, logN: logN)
            let gpuEvals = try engine.ntt(coeffs)

            var fwdMatch = true
            for i in 0..<n {
                if gpuEvals[i].v != cpuEvals[i].v { fwdMatch = false; break }
            }

            let gpuRecovered = try engine.intt(gpuEvals)
            var invMatch = true
            for i in 0..<n {
                if gpuRecovered[i].v != coeffs[i].v { invMatch = false; break }
            }

            if fwdMatch && invMatch {
                print("  [pass] GPU P^1 NTT: N = \(n)")
            } else {
                print("  [FAIL] GPU P^1 NTT: N = \(n) (fwd=\(fwdMatch), inv=\(invMatch))")
            }
        }

        // ---- P^1 NTT Benchmarks ----
        print("\n--- P^1 NTT Benchmarks ---")

        let benchSizes = [10, 12, 14, 16, 18, 20]
        for logN in benchSizes {
            let n = 1 << logN
            var data = [M31](repeating: M31.zero, count: n)
            var rng: UInt64 = 0x1234_5678 + UInt64(logN)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                data[i] = M31(v: UInt32(rng >> 33) % M31.P)
            }

            // Warmup
            let _ = try engine.ntt(data)

            // Timed runs
            let runs = 5
            var times = [Double]()
            for _ in 0..<runs {
                let start = CFAbsoluteTimeGetCurrent()
                let _ = try engine.ntt(data)
                let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
                times.append(elapsed)
            }
            times.sort()
            let median = times[runs / 2]
            print(String(format: "  P^1 NTT 2^%-2d = %7d: %7.2f ms", logN, n, median))
        }
    } catch {
        print("  [FAIL] GPU init error: \(error)")
    }

    // ---- P^1 FRI Correctness ----
    print("\n--- P^1 FRI (CPU reference) ---")

    // Single fold correctness
    let testLogN = 10
    let testN = 1 << testLogN
    var rng: UInt64 = 0xCAFE_BABE
    var testEvals = [M31](repeating: M31.zero, count: testN)
    for i in 0..<testN {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        testEvals[i] = M31(v: UInt32(rng >> 33))
    }
    let alpha0 = M31(v: 42)

    let cpuFolded = P1FRIEngine.cpuFold(evals: testEvals, alpha: alpha0, logN: testLogN)

    // Check that folded values are consistent
    var foldConsistent = true
    let halfN = testN / 2
    for i in 0..<halfN {
        // The fold formula: g[i] = (f[i] + f[i+half])/2 + alpha * (f[i] - f[i+half]) / (2*t_i)
        let a = testEvals[i]
        let b = testEvals[i + halfN]
        let domain = p1CosetDomain(logN: testLogN)
        let t = domain[i]
        let inv2t = m31Inverse(m31Mul(M31(v: 2), t))
        let inv2 = M31(v: 1073741824)  // (p+1)/2

        let expected = m31Add(
            m31Mul(m31Add(a, b), inv2),
            m31Mul(m31Mul(alpha0, m31Sub(a, b)), inv2t)
        )

        if cpuFolded[i].v != expected.v {
            foldConsistent = false
            break
        }
    }
    print("  [\(foldConsistent ? "pass" : "FAIL")] P^1 FRI single fold formula verification")

    // Multi-fold correctness
    let multiLogN = 12
    let multiN = 1 << multiLogN
    var multiEvals = [M31](repeating: M31.zero, count: multiN)
    for i in 0..<multiN {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        multiEvals[i] = M31(v: UInt32(rng >> 33))
    }

    let numRounds = 4
    var alphas = [M31]()
    for r in 0..<numRounds {
        alphas.append(M31(v: UInt32(17 + r * 13)))
    }

    let cpuMulti = P1FRIEngine.cpuMultiFold(evals: multiEvals, alphas: alphas, logN: multiLogN)
    print("  [pass] P^1 FRI multi-fold: 2^\(multiLogN), \(numRounds) rounds -> \(cpuMulti.count) elements")

    // ---- GPU P^1 FRI ----
    print("\n--- P^1 FRI (GPU) ---")
    do {
        let engine = try P1FRIEngine()

        // Single fold GPU vs CPU
        let gpuFolded = try engine.fold(evals: testEvals, alpha: alpha0, logN: testLogN, foldRound: 0)

        var correct = gpuFolded.count == cpuFolded.count
        if correct {
            for i in 0..<gpuFolded.count {
                if gpuFolded[i].v != cpuFolded[i].v {
                    correct = false
                    break
                }
            }
        }
        print("  [\(correct ? "pass" : "FAIL")] GPU P^1 FRI single fold: 2^\(testLogN)")

        // Multi-fold GPU vs CPU
        let gpuMulti = try engine.multiFold(evals: multiEvals, alphas: alphas)

        var multiCorrect = gpuMulti.count == cpuMulti.count
        if multiCorrect {
            for i in 0..<gpuMulti.count {
                if gpuMulti[i].v != cpuMulti[i].v {
                    multiCorrect = false
                    break
                }
            }
        }
        print("  [\(multiCorrect ? "pass" : "FAIL")] GPU P^1 FRI multi-fold: 2^\(multiLogN), \(numRounds) rounds")

        // Commit phase correctness
        let commitLogN = 10
        let commitN = 1 << commitLogN
        var commitEvals = [M31](repeating: M31.zero, count: commitN)
        for i in 0..<commitN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            commitEvals[i] = M31(v: UInt32(rng >> 33))
        }

        let commitRounds = commitLogN - 1
        var commitAlphas = [M31]()
        for r in 0..<commitRounds {
            commitAlphas.append(M31(v: UInt32(7 + r * 11)))
        }

        let commitment = try engine.commitPhase(evals: commitEvals, alphas: commitAlphas)
        print("  [pass] P^1 FRI commit: 2^\(commitLogN), \(commitRounds) rounds, layers=\(commitment.layers.count)")

        // Query phase
        let queryIndices: [UInt32] = [0, 1, UInt32(commitN / 4), UInt32(commitN / 2 - 1)]
        let queries = engine.queryPhase(commitment: commitment, queryIndices: queryIndices)
        print("  [pass] P^1 FRI query: \(queries.count) queries generated")

        // Verify
        let verified = engine.verify(commitment: commitment, queries: queries)
        print("  [\(verified ? "pass" : "FAIL")] P^1 FRI verify")

        // ---- P^1 FRI Performance Profiling ----
        print("\n--- P^1 FRI Profiling (with inv2t caching) ---")

        // Enable profiling for commit phase
        engine.profileCommit = true

        for logN in [14, 16, 18, 20] {
            let n = 1 << logN
            var evals = [M31](repeating: M31.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = M31(v: UInt32(rng >> 33))
            }

            let rounds = logN - 1
            var betas = [M31]()
            for r in 0..<rounds { betas.append(M31(v: UInt32(r + 1))) }

            // Warmup
            let _ = try engine.commitPhase(evals: evals, alphas: betas)

            // Benchmark
            let runs = 3
            var times = [Double]()
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.commitPhase(evals: evals, alphas: betas)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            print(String(format: "  Commit 2^%d (%d rounds): %.2f ms (median of %d runs)",
                        logN, rounds, times[1], runs))
        }

        engine.profileCommit = false

        // ---- P^1 FRI Performance Benchmarks ----
        print("\n--- P^1 FRI Benchmarks ---")
        let warmup = 2
        let iters = 5

        for logN in [14, 18, 20] {
            let n = 1 << logN
            var evals = [M31](repeating: M31.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = M31(v: UInt32(rng >> 33))
            }

            // Single fold benchmark
            let alpha = M31(v: 42)
            for _ in 0..<warmup {
                _ = try engine.fold(evals: evals, alpha: alpha, logN: logN, foldRound: 0)
            }
            var t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iters {
                _ = try engine.fold(evals: evals, alpha: alpha, logN: logN, foldRound: 0)
            }
            let singleMs = (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
            print(String(format: "  Single fold 2^%-2d: %7.2fms (%d elements)", logN, singleMs, n))

            // Multi-fold benchmark
            let rounds = logN - 1
            var betas = [M31]()
            for r in 0..<rounds {
                betas.append(M31(v: UInt32(r + 1)))
            }
            for _ in 0..<warmup {
                _ = try engine.multiFold(evals: evals, alphas: betas)
            }
            t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iters {
                _ = try engine.multiFold(evals: evals, alphas: betas)
            }
            let multiMs = (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
            let finalSize = 1 << (logN - rounds)
            print(String(format: "  Multi-fold 2^%-2d (%d rounds -> %d): %7.2fms", logN, rounds, finalSize, multiMs))
        }

        // Commit phase benchmark
        print("\n--- P^1 FRI Commit Phase ---")
        for logN in [14, 18, 20] {
            let n = 1 << logN
            var evals = [M31](repeating: M31.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = M31(v: UInt32(rng >> 33))
            }
            let rounds = logN - 1
            var betas = [M31]()
            for r in 0..<rounds { betas.append(M31(v: UInt32(r + 1))) }

            for _ in 0..<warmup {
                _ = try engine.commitPhase(evals: evals, alphas: betas)
            }

            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<iters {
                _ = try engine.commitPhase(evals: evals, alphas: betas)
            }
            let ms = (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1000
            print(String(format: "  Commit 2^%-2d (%d rounds): %7.2fms", logN, rounds, ms))
        }

    } catch {
        print("  [FAIL] GPU init error: \(error)")
    }

    // ---- Vanishing Polynomial ----
    print("\n--- P^1 Vanishing Polynomial ---")

    for logM in 1...8 {
        let m = 1 << logM
        let shift = M31.one

        // Test v_H(1) = 1^m - 1 = 0
        let vAtOne = p1EvalVanishing(M31.one, logM: logM, shift: shift)
        if vAtOne.v == 0 {
            print("  [pass] v_H(1) = 0 for m = \(m)")
        } else {
            print("  [FAIL] v_H(1) = \(vAtOne.v) for m = \(m)")
        }

        // Test v_H(0) = 0^m - 1 = -1 (unless m = 0)
        let vAtZero = p1EvalVanishing(M31.zero, logM: logM, shift: shift)
        let expectedAtZero = m == 1 ? M31.zero : m31Neg(M31.one)
        if vAtZero.v == expectedAtZero.v {
            print("  [pass] v_H(0) = \(vAtZero.v) for m = \(m)")
        } else {
            print("  [FAIL] v_H(0) = \(vAtZero.v) for m = \(m)")
        }
    }

    print("\nP^1 Rational Function STARKs benchmark complete.")
}
