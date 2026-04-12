import zkMetal
import Foundation

func runPastaNTTTests() {
    // ============================================================
    // Pallas Fr NTT tests
    // ============================================================

    suite("Pallas Fr C NTT")
    do {
        // Round-trip: NTT -> INTT = identity (small sizes)
        for logN in 2...10 {
            let n = 1 << logN
            var input = [VestaFp](repeating: VestaFp.zero, count: n)
            for i in 0..<n { input[i] = vestaFromInt(UInt64(i + 1)) }
            let fwd = pallasFrCpuNTT(input, logN: logN)
            let rec = pallasFrCpuINTT(fwd, logN: logN)
            var ok = true
            for i in 0..<n {
                if vestaToInt(input[i]) != vestaToInt(rec[i]) { ok = false; break }
            }
            expect(ok, "Pallas Fr C round-trip N=\(n)")
        }

        // Naive DFT cross-check at small sizes
        for logN in 2...4 {
            let n = 1 << logN
            var input = [VestaFp](repeating: VestaFp.zero, count: n)
            for i in 0..<n { input[i] = vestaFromInt(UInt64(i + 1)) }

            let nttResult = pallasFrCpuNTT(input, logN: logN)

            // Naive DFT: Y[k] = sum_j input[j] * omega^(j*k)
            // omega is the primitive n-th root of unity
            // We compute omega = root_of_unity^(2^(32-logN))
            // But since we don't expose root directly, let's verify via polynomial evaluation:
            // If input = [a0, a1, ..., a_{n-1}], then NTT[k] = sum a_j * omega^(jk)
            // This is poly(omega^k) where poly(x) = a0 + a1*x + ... + a_{n-1}*x^{n-1}

            // Simple check: NTT[0] = sum of all inputs (omega^0 = 1)
            var sum = VestaFp.zero
            for i in 0..<n { sum = vestaAdd(sum, input[i]) }
            expect(vestaToInt(sum) == vestaToInt(nttResult[0]),
                   "Pallas Fr NTT[0]=sum N=\(n)")
        }

        // Convolution test: NTT-based polynomial multiplication
        do {
            let logN = 4
            let n = 1 << logN
            // poly a = 1 + 2x + 3x^2
            var a = [VestaFp](repeating: VestaFp.zero, count: n)
            a[0] = vestaFromInt(1); a[1] = vestaFromInt(2); a[2] = vestaFromInt(3)
            // poly b = 1 + x
            var b = [VestaFp](repeating: VestaFp.zero, count: n)
            b[0] = vestaFromInt(1); b[1] = vestaFromInt(1)

            let fa = pallasFrCpuNTT(a, logN: logN)
            let fb = pallasFrCpuNTT(b, logN: logN)
            var fc = [VestaFp](repeating: VestaFp.zero, count: n)
            for i in 0..<n { fc[i] = vestaMul(fa[i], fb[i]) }
            let c = pallasFrCpuINTT(fc, logN: logN)

            // Expected: (1+2x+3x^2)(1+x) = 1 + 3x + 5x^2 + 3x^3
            let cInt = (0..<n).map { vestaToInt(c[$0]) }
            expect(cInt[0] == [1, 0, 0, 0], "Pallas poly mul c[0]=1")
            expect(cInt[1] == [3, 0, 0, 0], "Pallas poly mul c[1]=3")
            expect(cInt[2] == [5, 0, 0, 0], "Pallas poly mul c[2]=5")
            expect(cInt[3] == [3, 0, 0, 0], "Pallas poly mul c[3]=3")
            for i in 4..<n {
                expect(cInt[i] == [0, 0, 0, 0], "Pallas poly mul c[\(i)]=0")
            }
        }

        // Large round-trip test
        do {
            let logN = 16
            let n = 1 << logN
            var input = [VestaFp](repeating: VestaFp.zero, count: n)
            var rng: UInt64 = 0xDEAD_BEEF
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = vestaFromInt(rng >> 32)
            }
            let fwd = pallasFrCpuNTT(input, logN: logN)
            let rec = pallasFrCpuINTT(fwd, logN: logN)
            var mm = 0
            for i in 0..<n {
                if vestaToInt(input[i]) != vestaToInt(rec[i]) { mm += 1 }
            }
            expect(mm == 0, "Pallas Fr C round-trip 2^16")
        }
    }

    suite("Pallas Fr GPU NTT")
    do {
        let engine = try PallasNTTEngine()

        // GPU round-trip 2^10
        do {
            let n = 1024
            var input = [VestaFp](repeating: VestaFp.zero, count: n)
            for i in 0..<n { input[i] = vestaFromInt(UInt64(i + 1)) }
            let fwd = try engine.ntt(input)
            let rec = try engine.intt(fwd)
            var ok = true
            for i in 0..<n {
                if vestaToInt(input[i]) != vestaToInt(rec[i]) { ok = false; break }
            }
            expect(ok, "Pallas GPU round-trip 2^10")
        }

        // GPU vs CPU cross-check 2^10
        do {
            let n = 1024
            var input = [VestaFp](repeating: VestaFp.zero, count: n)
            for i in 0..<n { input[i] = vestaFromInt(UInt64(i + 1)) }
            let gpuFwd = try engine.ntt(input)
            let cpuFwd = pallasFrCpuNTT(input, logN: 10)
            var ok = true
            for i in 0..<n {
                if vestaToInt(gpuFwd[i]) != vestaToInt(cpuFwd[i]) { ok = false; break }
            }
            expect(ok, "Pallas GPU vs CPU 2^10")
        }

        // GPU round-trip 2^14
        do {
            let n = 1 << 14
            var input = [VestaFp](repeating: VestaFp.zero, count: n)
            var rng: UInt64 = 0xABCD_1234
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = vestaFromInt(rng >> 32)
            }
            let fwd = try engine.ntt(input)
            let rec = try engine.intt(fwd)
            var mm = 0
            for i in 0..<n {
                if vestaToInt(input[i]) != vestaToInt(rec[i]) { mm += 1 }
            }
            expect(mm == 0, "Pallas GPU round-trip 2^14")
        }
    } catch {
        expect(false, "Pallas GPU NTT error: \(error)")
    }

    // ============================================================
    // Vesta Fr NTT tests
    // ============================================================

    suite("Vesta Fr C NTT")
    do {
        // Round-trip: NTT -> INTT = identity (small sizes)
        for logN in 2...10 {
            let n = 1 << logN
            var input = [PallasFp](repeating: PallasFp.zero, count: n)
            for i in 0..<n { input[i] = pallasFromInt(UInt64(i + 1)) }
            let fwd = vestaFrCpuNTT(input, logN: logN)
            let rec = vestaFrCpuINTT(fwd, logN: logN)
            var ok = true
            for i in 0..<n {
                if pallasToInt(input[i]) != pallasToInt(rec[i]) { ok = false; break }
            }
            expect(ok, "Vesta Fr C round-trip N=\(n)")
        }

        // NTT[0] = sum of all inputs
        for logN in 2...4 {
            let n = 1 << logN
            var input = [PallasFp](repeating: PallasFp.zero, count: n)
            for i in 0..<n { input[i] = pallasFromInt(UInt64(i + 1)) }
            let nttResult = vestaFrCpuNTT(input, logN: logN)
            var sum = PallasFp.zero
            for i in 0..<n { sum = pallasAdd(sum, input[i]) }
            expect(pallasToInt(sum) == pallasToInt(nttResult[0]),
                   "Vesta Fr NTT[0]=sum N=\(n)")
        }

        // Convolution test: NTT-based polynomial multiplication
        do {
            let logN = 4
            let n = 1 << logN
            var a = [PallasFp](repeating: PallasFp.zero, count: n)
            a[0] = pallasFromInt(1); a[1] = pallasFromInt(2); a[2] = pallasFromInt(3)
            var b = [PallasFp](repeating: PallasFp.zero, count: n)
            b[0] = pallasFromInt(1); b[1] = pallasFromInt(1)

            let fa = vestaFrCpuNTT(a, logN: logN)
            let fb = vestaFrCpuNTT(b, logN: logN)
            var fc = [PallasFp](repeating: PallasFp.zero, count: n)
            for i in 0..<n { fc[i] = pallasMul(fa[i], fb[i]) }
            let c = vestaFrCpuINTT(fc, logN: logN)

            let cInt = (0..<n).map { pallasToInt(c[$0]) }
            expect(cInt[0] == [1, 0, 0, 0], "Vesta poly mul c[0]=1")
            expect(cInt[1] == [3, 0, 0, 0], "Vesta poly mul c[1]=3")
            expect(cInt[2] == [5, 0, 0, 0], "Vesta poly mul c[2]=5")
            expect(cInt[3] == [3, 0, 0, 0], "Vesta poly mul c[3]=3")
            for i in 4..<n {
                expect(cInt[i] == [0, 0, 0, 0], "Vesta poly mul c[\(i)]=0")
            }
        }

        // Large round-trip
        do {
            let logN = 16
            let n = 1 << logN
            var input = [PallasFp](repeating: PallasFp.zero, count: n)
            var rng: UInt64 = 0xFACE_CAFE
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = pallasFromInt(rng >> 32)
            }
            let fwd = vestaFrCpuNTT(input, logN: logN)
            let rec = vestaFrCpuINTT(fwd, logN: logN)
            var mm = 0
            for i in 0..<n {
                if pallasToInt(input[i]) != pallasToInt(rec[i]) { mm += 1 }
            }
            expect(mm == 0, "Vesta Fr C round-trip 2^16")
        }
    }

    suite("Vesta Fr GPU NTT")
    do {
        let engine = try VestaNTTEngine()

        // GPU round-trip 2^10
        do {
            let n = 1024
            var input = [PallasFp](repeating: PallasFp.zero, count: n)
            for i in 0..<n { input[i] = pallasFromInt(UInt64(i + 1)) }
            let fwd = try engine.ntt(input)
            let rec = try engine.intt(fwd)
            var ok = true
            for i in 0..<n {
                if pallasToInt(input[i]) != pallasToInt(rec[i]) { ok = false; break }
            }
            expect(ok, "Vesta GPU round-trip 2^10")
        }

        // GPU vs CPU cross-check 2^10
        do {
            let n = 1024
            var input = [PallasFp](repeating: PallasFp.zero, count: n)
            for i in 0..<n { input[i] = pallasFromInt(UInt64(i + 1)) }
            let gpuFwd = try engine.ntt(input)
            let cpuFwd = vestaFrCpuNTT(input, logN: 10)
            var ok = true
            for i in 0..<n {
                if pallasToInt(gpuFwd[i]) != pallasToInt(cpuFwd[i]) { ok = false; break }
            }
            expect(ok, "Vesta GPU vs CPU 2^10")
        }

        // GPU round-trip 2^14
        do {
            let n = 1 << 14
            var input = [PallasFp](repeating: PallasFp.zero, count: n)
            var rng: UInt64 = 0x1234_5678
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = pallasFromInt(rng >> 32)
            }
            let fwd = try engine.ntt(input)
            let rec = try engine.intt(fwd)
            var mm = 0
            for i in 0..<n {
                if pallasToInt(input[i]) != pallasToInt(rec[i]) { mm += 1 }
            }
            expect(mm == 0, "Vesta GPU round-trip 2^14")
        }
    } catch {
        expect(false, "Vesta GPU NTT error: \(error)")
    }
}

// MARK: - Benchmarks

private func pastaNTTBenchmark() {
    fputs("\n  Pasta NTT Benchmark:\n", stderr)

    do {
        let pallasEngine = try PallasNTTEngine()
        let vestaEngine = try VestaNTTEngine()

        // Pallas GPU benchmark
        fputs("  Pallas (GPU):\n", stderr)
        for logN in [12, 14, 16, 18] {
            let n = 1 << logN
            var input = [VestaFp](repeating: VestaFp.zero, count: n)
            var rng: UInt64 = 0x1234_5678
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = vestaFromInt(rng >> 32)
            }

            var times = [Double]()
            for _ in 0..<3 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let fwd = try pallasEngine.ntt(input)
                let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                times.append(dt)
            }
            times.sort()
            fputs(String(format: "    2^%d: %.2f ms\n", logN, times[1]), stderr)
        }

        // Vesta GPU benchmark
        fputs("  Vesta (GPU):\n", stderr)
        for logN in [12, 14, 16, 18] {
            let n = 1 << logN
            var input = [PallasFp](repeating: PallasFp.zero, count: n)
            var rng: UInt64 = 0x1234_5678
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = pallasFromInt(rng >> 32)
            }

            var times = [Double]()
            for _ in 0..<3 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let fwd = try vestaEngine.ntt(input)
                let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                times.append(dt)
            }
            times.sort()
            fputs(String(format: "    2^%d: %.2f ms\n", logN, times[1]), stderr)
        }
    } catch {
        fputs("  Pasta NTT benchmark error: \(error)\n", stderr)
    }
}

public func runPastaNTTBenchmark() {
    pastaNTTBenchmark()
}
