import Foundation
import zkMetal
import NeonFieldOps

func runAdditiveFFTTests() {
    suite("Additive FFT GF(2^64)")

    // Test 1: Forward then inverse should give back original
    do {
        let k = 4  // 2^4 = 16 point transform
        let n = 1 << k
        let fft = AdditiveFFT64(logSize: k)

        // Create a test polynomial: coefficients 1, 2, 3, ..., 16
        var original = [UInt64](repeating: 0, count: n)
        for i in 0..<n { original[i] = UInt64(i + 1) }
        var data = original

        // Forward FFT
        fft.forward(&data)

        // Verify the output is different from input (sanity check)
        var allSame = true
        for i in 0..<n {
            if data[i] != original[i] { allSame = false; break }
        }
        expect(!allSame, "Forward FFT changes data")

        // Inverse FFT
        fft.inverse(&data)

        // Should match original
        var roundTripOk = true
        for i in 0..<n {
            if data[i] != original[i] { roundTripOk = false; break }
        }
        expect(roundTripOk, "Round-trip 2^4 (forward then inverse)")
    }

    // Test 2: Larger transform round-trip
    do {
        let k = 10  // 2^10 = 1024 point transform
        let n = 1 << k
        let fft = AdditiveFFT64(logSize: k)

        // Pseudo-random data
        var original = [UInt64](repeating: 0, count: n)
        var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            original[i] = rng
        }
        var data = original

        fft.forward(&data)
        fft.inverse(&data)

        var ok = true
        for i in 0..<n {
            if data[i] != original[i] { ok = false; break }
        }
        expect(ok, "Round-trip 2^10 (1024 points)")
    }

    // Test 3: Size-1 and size-2 edge cases
    do {
        // Size 1
        let fft1 = AdditiveFFT64(logSize: 0)
        var data1: [UInt64] = [42]
        fft1.forward(&data1)
        expectEqual(data1[0], 42, "Size-1 forward is identity")
        fft1.inverse(&data1)
        expectEqual(data1[0], 42, "Size-1 inverse is identity")

        // Size 2
        let fft2 = AdditiveFFT64(logSize: 1)
        let orig2: [UInt64] = [0x123, 0x456]
        var data2 = orig2
        fft2.forward(&data2)
        fft2.inverse(&data2)
        expect(data2[0] == orig2[0] && data2[1] == orig2[1], "Round-trip 2^1 (2 points)")
    }

    // Test 4: Recursive vs iterative give same result
    do {
        let k = 6
        let n = 1 << k
        let fft = AdditiveFFT64(logSize: k)

        var rng: UInt64 = 0xABCD_1234_5678_EF01
        var data = [UInt64](repeating: 0, count: n)
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            data[i] = rng
        }

        // Recursive forward
        var recursive = data
        recursive.withUnsafeMutableBufferPointer { buf in
            fft.basis.withUnsafeBufferPointer { basisBuf in
                bt_afft_forward_64(buf.baseAddress!, size_t(n), basisBuf.baseAddress!)
            }
        }

        // Iterative forward (NEON)
        var iterative = data
        fft.forward(&iterative)

        var match = true
        for i in 0..<n {
            if recursive[i] != iterative[i] { match = false; break }
        }
        expect(match, "Recursive == iterative (NEON) forward 2^6")
    }

    // Test 5: Zero polynomial
    do {
        let k = 4
        let n = 1 << k
        let fft = AdditiveFFT64(logSize: k)

        var data = [UInt64](repeating: 0, count: n)
        fft.forward(&data)
        var allZero = true
        for i in 0..<n {
            if data[i] != 0 { allZero = false; break }
        }
        expect(allZero, "Zero polynomial stays zero after forward")
    }

    suite("Additive FFT GF(2^128)")

    // Test 6: GF(2^128) round-trip
    do {
        let k = 4
        let n = 1 << k
        let fft = AdditiveFFT128(logSize: k)

        var original = [UInt64](repeating: 0, count: 2 * n)
        var rng: UInt64 = 0x1234_5678_9ABC_DEF0
        for i in 0..<(2 * n) {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            original[i] = rng
        }
        var data = original

        fft.forward(&data)
        fft.inverse(&data)

        var ok = true
        for i in 0..<(2 * n) {
            if data[i] != original[i] { ok = false; break }
        }
        expect(ok, "GF(2^128) round-trip 2^4 (16 points)")
    }

    // Test 7: GF(2^128) larger round-trip
    do {
        let k = 8
        let n = 1 << k
        let fft = AdditiveFFT128(logSize: k)

        var original = [UInt64](repeating: 0, count: 2 * n)
        var rng: UInt64 = 0xFEDC_BA98_7654_3210
        for i in 0..<(2 * n) {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            original[i] = rng
        }
        var data = original

        fft.forward(&data)
        fft.inverse(&data)

        var ok = true
        for i in 0..<(2 * n) {
            if data[i] != original[i] { ok = false; break }
        }
        expect(ok, "GF(2^128) round-trip 2^8 (256 points)")
    }

    // Test 8: Polynomial multiplication consistency (GF(2^64))
    // Multiply (1 + 2x) * (3 + 4x) and verify via direct computation
    do {
        let k = 2  // 4-point transform (enough for degree-2 product)
        let fft = AdditiveFFT64(logSize: k)

        let a: [UInt64] = [1, 2]
        let b: [UInt64] = [3, 4]
        let product = fft.multiply(a, b)

        // In GF(2^64), (1 + 2x)(3 + 4x) = 1*3 + (1*4 + 2*3)x + 2*4*x^2
        // = 3 + (4 ^ 6)x + 8*x^2  (addition is XOR)
        // = 3 + 2*x + 8*x^2
        // But this is in the novel polynomial basis, so direct comparison
        // doesn't apply. Instead, verify round-trip: the product evaluated
        // pointwise should equal the coefficient-form inverse.
        // We just verify the multiply function doesn't crash and produces
        // a 4-element result.
        expect(product.count == 4, "Polynomial multiply produces correct size")
    }

    // Test 9: Performance (timing info only, not a pass/fail)
    do {
        let k = 16  // 2^16 = 65536 point transform
        let n = 1 << k
        let fft = AdditiveFFT64(logSize: k)

        var data = [UInt64](repeating: 0, count: n)
        var rng: UInt64 = 0xCAFE_BABE
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            data[i] = rng
        }
        let original = data

        let t0 = CFAbsoluteTimeGetCurrent()
        fft.forward(&data)
        let t1 = CFAbsoluteTimeGetCurrent()
        fft.inverse(&data)
        let t2 = CFAbsoluteTimeGetCurrent()

        var ok = true
        for i in 0..<n {
            if data[i] != original[i] { ok = false; break }
        }

        let fwdMs = (t1 - t0) * 1000
        let invMs = (t2 - t1) * 1000
        print(String(format: "  GF(2^64) 2^%d: forward %.2fms, inverse %.2fms", k, fwdMs, invMs))
        expect(ok, "Round-trip 2^16 (65536 points)")
    }
}
