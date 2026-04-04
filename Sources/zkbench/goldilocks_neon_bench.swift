// Benchmark: NEON-optimized Goldilocks field operations vs scalar
import Foundation
import NeonFieldOps

public func runGoldilocksNeonBench() {
    fputs("\n=== Goldilocks NEON Benchmark ===\n", stderr)

    let GL_P: UInt64 = 0xFFFFFFFF00000001

    // --- RNG ---
    var rng: UInt64 = 0xCAFE_BABE_DEAD_BEEF
    func nextRand() -> UInt64 {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        return rng % GL_P
    }

    // --- Correctness Tests ---
    fputs("Correctness checks...\n", stderr)

    let testN = 1024
    var a = [UInt64](repeating: 0, count: testN)
    var b = [UInt64](repeating: 0, count: testN)
    for i in 0..<testN {
        a[i] = nextRand()
        b[i] = nextRand()
    }

    // Test batch add
    var outNeon = [UInt64](repeating: 0, count: testN)
    var outScalar = [UInt64](repeating: 0, count: testN)

    gl_batch_add_neon(a, b, &outNeon, Int32(testN))
    // Scalar reference: add individually
    for i in 0..<testN {
        var s = a[i] &+ b[i]
        let carry = s < a[i]
        if carry {
            s = s &+ 0xFFFFFFFF
            if s < 0xFFFFFFFF { s = s &+ 0xFFFFFFFF }
        }
        if s >= GL_P { s = s &- GL_P }
        outScalar[i] = s
    }
    var addOk = true
    for i in 0..<testN {
        if outNeon[i] != outScalar[i] {
            fputs("  FAIL: batch_add mismatch at \(i): neon=\(String(format: "%016llx", outNeon[i])) scalar=\(String(format: "%016llx", outScalar[i]))\n", stderr)
            addOk = false
            break
        }
    }
    fputs("  batch_add:  \(addOk ? "PASS" : "FAIL")\n", stderr)

    // Test batch sub
    gl_batch_sub_neon(a, b, &outNeon, Int32(testN))
    for i in 0..<testN {
        outScalar[i] = a[i] >= b[i] ? a[i] &- b[i] : a[i] &+ GL_P &- b[i]
    }
    var subOk = true
    for i in 0..<testN {
        if outNeon[i] != outScalar[i] {
            fputs("  FAIL: batch_sub mismatch at \(i): neon=\(String(format: "%016llx", outNeon[i])) scalar=\(String(format: "%016llx", outScalar[i]))\n", stderr)
            subOk = false
            break
        }
    }
    fputs("  batch_sub:  \(subOk ? "PASS" : "FAIL")\n", stderr)

    // Test batch mul (use NTT roundtrip for correctness)
    // NTT roundtrip: intt(ntt(x)) == x
    for logN in [8, 10, 12] {
        let n = 1 << logN
        var data = [UInt64](repeating: 0, count: n)
        for i in 0..<n { data[i] = nextRand() }
        let orig = data

        // Scalar NTT roundtrip
        var scalarData = data
        goldilocks_ntt(&scalarData, Int32(logN))
        goldilocks_intt(&scalarData, Int32(logN))
        var scalarOk = true
        for i in 0..<n {
            if scalarData[i] != orig[i] { scalarOk = false; break }
        }

        // NEON NTT roundtrip
        var neonData = data
        goldilocks_ntt_neon(&neonData, Int32(logN))
        goldilocks_intt_neon(&neonData, Int32(logN))
        var neonOk = true
        for i in 0..<n {
            if neonData[i] != orig[i] { neonOk = false; break }
        }

        // Cross-check: scalar NTT == NEON NTT
        var scalarFwd = data
        var neonFwd = data
        goldilocks_ntt(&scalarFwd, Int32(logN))
        goldilocks_ntt_neon(&neonFwd, Int32(logN))
        var crossOk = true
        for i in 0..<n {
            if scalarFwd[i] != neonFwd[i] { crossOk = false; break }
        }

        fputs("  NTT 2^\(logN): scalar_roundtrip=\(scalarOk ? "PASS" : "FAIL") neon_roundtrip=\(neonOk ? "PASS" : "FAIL") cross=\(crossOk ? "PASS" : "FAIL")\n", stderr)
    }

    // --- Performance Benchmarks ---
    fputs("\n--- Batch Operations ---\n", stderr)

    for logN in [14, 16, 18, 20] {
        let n = 1 << logN
        var aa = [UInt64](repeating: 0, count: n)
        var bb = [UInt64](repeating: 0, count: n)
        var out = [UInt64](repeating: 0, count: n)
        for i in 0..<n { aa[i] = nextRand(); bb[i] = nextRand() }

        // Batch add — scalar baseline
        let scalarAddT0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<5 {
            for i in 0..<n {
                var s = aa[i] &+ bb[i]
                if s < aa[i] { s = s &+ 0xFFFFFFFF }
                if s >= GL_P { s = s &- GL_P }
                out[i] = s
            }
        }
        let scalarAddTime = (CFAbsoluteTimeGetCurrent() - scalarAddT0) * 1000 / 5

        // Batch add — NEON
        let neonAddT0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<5 {
            gl_batch_add_neon(aa, bb, &out, Int32(n))
        }
        let neonAddTime = (CFAbsoluteTimeGetCurrent() - neonAddT0) * 1000 / 5

        // Batch mul — scalar baseline
        let scalarMulT0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<5 {
            for i in 0..<n {
                out[i] = aa[i] &* bb[i]  // placeholder, not correct but measures loop overhead
            }
        }
        let scalarMulLoop = (CFAbsoluteTimeGetCurrent() - scalarMulT0) * 1000 / 5

        // Batch mul — NEON (interleaved scalar)
        let neonMulT0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<5 {
            gl_batch_mul_neon(aa, bb, &out, Int32(n))
        }
        let neonMulTime = (CFAbsoluteTimeGetCurrent() - neonMulT0) * 1000 / 5

        fputs(String(format: "  2^%-2d add: scalar %.2f ms, NEON %.2f ms (%.1f×) | mul: NEON %.2f ms\n",
                    logN, scalarAddTime, neonAddTime, scalarAddTime / max(neonAddTime, 0.001), neonMulTime), stderr)
    }

    // --- NTT Benchmark ---
    fputs("\n--- NTT Benchmark ---\n", stderr)

    for logN in [10, 12, 14, 16, 18, 20] {
        let n = 1 << logN
        var data = [UInt64](repeating: 0, count: n)
        for i in 0..<n { data[i] = nextRand() }

        let runs = logN <= 14 ? 20 : 5

        // Warmup
        var scratch = data
        goldilocks_ntt(&scratch, Int32(logN))
        scratch = data
        goldilocks_ntt_neon(&scratch, Int32(logN))

        // Scalar NTT
        var scalarTimes = [Double]()
        for _ in 0..<runs {
            scratch = data
            let t0 = CFAbsoluteTimeGetCurrent()
            goldilocks_ntt(&scratch, Int32(logN))
            scalarTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        scalarTimes.sort()
        let scalarMedian = scalarTimes[runs / 2]

        // NEON NTT
        var neonTimes = [Double]()
        for _ in 0..<runs {
            scratch = data
            let t0 = CFAbsoluteTimeGetCurrent()
            goldilocks_ntt_neon(&scratch, Int32(logN))
            neonTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        neonTimes.sort()
        let neonMedian = neonTimes[runs / 2]

        let speedup = scalarMedian / max(neonMedian, 0.001)
        fputs(String(format: "  NTT 2^%-2d: scalar %8.3f ms | NEON %8.3f ms | %.2f×\n",
                    logN, scalarMedian, neonMedian, speedup), stderr)
    }

    // --- Inverse NTT Benchmark ---
    fputs("\n--- Inverse NTT Benchmark ---\n", stderr)

    for logN in [10, 12, 14, 16, 18] {
        let n = 1 << logN
        var data = [UInt64](repeating: 0, count: n)
        for i in 0..<n { data[i] = nextRand() }

        // Forward NTT first
        goldilocks_ntt_neon(&data, Int32(logN))

        let runs = logN <= 14 ? 20 : 5
        var scratch = data

        // Warmup
        goldilocks_intt(&scratch, Int32(logN))
        scratch = data
        goldilocks_intt_neon(&scratch, Int32(logN))

        // Scalar INTT
        var scalarTimes = [Double]()
        for _ in 0..<runs {
            scratch = data
            let t0 = CFAbsoluteTimeGetCurrent()
            goldilocks_intt(&scratch, Int32(logN))
            scalarTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        scalarTimes.sort()
        let scalarMedian = scalarTimes[runs / 2]

        // NEON INTT
        var neonTimes = [Double]()
        for _ in 0..<runs {
            scratch = data
            let t0 = CFAbsoluteTimeGetCurrent()
            goldilocks_intt_neon(&scratch, Int32(logN))
            neonTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }
        neonTimes.sort()
        let neonMedian = neonTimes[runs / 2]

        let speedup = scalarMedian / max(neonMedian, 0.001)
        fputs(String(format: "  INTT 2^%-2d: scalar %8.3f ms | NEON %8.3f ms | %.2f×\n",
                    logN, scalarMedian, neonMedian, speedup), stderr)
    }

    fputs("\nDone.\n", stderr)
}
