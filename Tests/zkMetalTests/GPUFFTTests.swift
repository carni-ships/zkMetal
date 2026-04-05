import Foundation
import zkMetal

public func runGPUFFTTests() {
    suite("GPU FFT Engine - BN254 Fr")

    do {
        let engine = try GPUFFTEngine()
        let nttEngine = try NTTEngine()

        // Test 1: FFT matches CPU reference (NTTEngine.cpuNTT) for small sizes
        for logN in [4, 8] {
            let n = 1 << logN
            var input = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n { input[i] = frFromInt(UInt64(i + 1)) }

            let gpuResult = try engine.fft(data: input, logN: logN, inverse: false)
            let cpuResult = NTTEngine.cpuNTT(input, logN: logN)

            var ok = true
            for i in 0..<n {
                if frToInt(gpuResult[i]) != frToInt(cpuResult[i]) { ok = false; break }
            }
            expect(ok, "FFT matches CPU reference 2^\(logN)")
        }

        // Test 2: FFT matches CPU reference for medium sizes
        for logN in [12] {
            let n = 1 << logN
            var input = [Fr](repeating: Fr.zero, count: n)
            var rng: UInt64 = 0xDEAD_BEEF
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = frFromInt(rng >> 32)
            }

            let gpuResult = try engine.fft(data: input, logN: logN, inverse: false)
            let cpuResult = NTTEngine.cpuNTT(input, logN: logN)

            var ok = true
            for i in 0..<n {
                if frToInt(gpuResult[i]) != frToInt(cpuResult[i]) { ok = false; break }
            }
            expect(ok, "FFT matches CPU reference 2^\(logN)")
        }

        // Test 3: Inverse FFT recovers original for various sizes
        for logN in [4, 8, 12] {
            let n = 1 << logN
            var input = [Fr](repeating: Fr.zero, count: n)
            var rng: UInt64 = 0xCAFE_0000 + UInt64(logN)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = frFromInt(rng >> 32)
            }

            let fwd = try engine.fft(data: input, logN: logN, inverse: false)
            let rec = try engine.fft(data: fwd, logN: logN, inverse: true)

            var ok = true
            for i in 0..<n {
                if frToInt(input[i]) != frToInt(rec[i]) { ok = false; break }
            }
            expect(ok, "IFFT recovers original 2^\(logN)")
        }

        // Test 4: GPU FFT matches existing NTTEngine GPU output
        for logN in [8, 12] {
            let n = 1 << logN
            var input = [Fr](repeating: Fr.zero, count: n)
            var rng: UInt64 = 0xBEEF_CAFE + UInt64(logN)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = frFromInt(rng >> 32)
            }

            let fftResult = try engine.fft(data: input, logN: logN, inverse: false)
            let nttResult = try nttEngine.ntt(input)

            var ok = true
            for i in 0..<n {
                if frToInt(fftResult[i]) != frToInt(nttResult[i]) { ok = false; break }
            }
            expect(ok, "GPU FFT matches NTTEngine 2^\(logN)")
        }

        // Test 5: Convolution via FFT: FFT(a*b) = FFT(a) . FFT(b)
        do {
            // Polynomial a(x) = 1 + 2x + 3x^2
            // Polynomial b(x) = 4 + 5x
            // Product: 4 + 13x + 22x^2 + 15x^3
            let a = [frFromInt(1), frFromInt(2), frFromInt(3)]
            let b = [frFromInt(4), frFromInt(5)]

            let result = try engine.convolve(a, b)

            // Check coefficients (result is length 8 due to padding)
            let c0 = frToInt(result[0])
            let c1 = frToInt(result[1])
            let c2 = frToInt(result[2])
            let c3 = frToInt(result[3])

            let expected0 = frToInt(frFromInt(4))
            let expected1 = frToInt(frFromInt(13))
            let expected2 = frToInt(frFromInt(22))
            let expected3 = frToInt(frFromInt(15))

            let convOk = (c0 == expected0) && (c1 == expected1) &&
                         (c2 == expected2) && (c3 == expected3)
            expect(convOk, "Convolution via FFT: (1+2x+3x^2)*(4+5x)")

            // Remaining coefficients should be zero
            var zeroOk = true
            for i in 4..<result.count {
                if !frToInt(result[i]).allSatisfy({ $0 == 0 }) { zeroOk = false; break }
            }
            expect(zeroOk, "Convolution high terms are zero")
        }

        // Test 6: Larger round-trip 2^16
        do {
            let logN = 16
            let n = 1 << logN
            var input = [Fr](repeating: Fr.zero, count: n)
            var rng: UInt64 = 0x1234_5678
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = frFromInt(rng >> 32)
            }

            let fwd = try engine.fft(data: input, logN: logN, inverse: false)
            let rec = try engine.fft(data: fwd, logN: logN, inverse: true)

            var mismatches = 0
            for i in 0..<n {
                if frToInt(input[i]) != frToInt(rec[i]) { mismatches += 1 }
            }
            expect(mismatches == 0, "Round-trip 2^16 (mismatches: \(mismatches))")
        }

        // Test 7: Larger FFT vs CPU 2^16
        do {
            let logN = 16
            let n = 1 << logN
            var input = [Fr](repeating: Fr.zero, count: n)
            var rng: UInt64 = 0xAAAA_BBBB
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = frFromInt(rng >> 32)
            }

            let gpuResult = try engine.fft(data: input, logN: logN, inverse: false)
            let cpuResult = NTTEngine.cpuNTT(input, logN: logN)

            var mismatches = 0
            for i in 0..<n {
                if frToInt(gpuResult[i]) != frToInt(cpuResult[i]) { mismatches += 1 }
            }
            expect(mismatches == 0, "FFT matches CPU 2^16 (mismatches: \(mismatches))")
        }

        // Test 8: Performance comparison vs NTTEngine (2^16)
        do {
            let logN = 16
            let n = 1 << logN
            var input = [Fr](repeating: Fr.zero, count: n)
            var rng: UInt64 = 0xFACE_B00C
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = frFromInt(rng >> 32)
            }

            // Warm up
            _ = try engine.fft(data: input, logN: logN, inverse: false)
            _ = try nttEngine.ntt(input)

            let iters = 5

            var fftTotal: Double = 0
            for _ in 0..<iters {
                let start = CFAbsoluteTimeGetCurrent()
                _ = try engine.fft(data: input, logN: logN, inverse: false)
                fftTotal += (CFAbsoluteTimeGetCurrent() - start) * 1000.0
            }

            var nttTotal: Double = 0
            for _ in 0..<iters {
                let start = CFAbsoluteTimeGetCurrent()
                _ = try nttEngine.ntt(input)
                nttTotal += (CFAbsoluteTimeGetCurrent() - start) * 1000.0
            }

            let fftAvg = fftTotal / Double(iters)
            let nttAvg = nttTotal / Double(iters)
            print("  [PERF] GPUFFTEngine 2^16: \(String(format: "%.3f", fftAvg)) ms")
            print("  [PERF] NTTEngine    2^16: \(String(format: "%.3f", nttAvg)) ms")
            // Just verify it completes - no strict performance requirement
            expect(true, "Performance benchmark completed")
        }

        // Test 9: Large round-trip 2^20
        do {
            let logN = 20
            let n = 1 << logN
            var input = [Fr](repeating: Fr.zero, count: n)
            var rng: UInt64 = 0x9876_5432
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                input[i] = frFromInt(rng >> 32)
            }

            let fwd = try engine.fft(data: input, logN: logN, inverse: false)
            let rec = try engine.fft(data: fwd, logN: logN, inverse: true)

            var mismatches = 0
            for i in 0..<n {
                if frToInt(input[i]) != frToInt(rec[i]) { mismatches += 1 }
            }
            expect(mismatches == 0, "Round-trip 2^20 (mismatches: \(mismatches))")
        }

    } catch {
        expect(false, "GPU FFT error: \(error)")
    }
}
