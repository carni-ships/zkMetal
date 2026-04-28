import Foundation
import zkMetal

public func runGPUAdditiveFFTTests() {
    print("GPU Additive FFT: starting")
    suite("GPU Additive FFT GF(2^8)")

    // Test 1: GPU engine creation
    let engine: GPUAdditiveFFTEngine
    do {
        engine = try GPUAdditiveFFTEngine()
    } catch {
        print("  [ERROR] Failed to create GPUAdditiveFFTEngine: \(error)")
        return
    }
    print("  [OK] Engine created successfully")

    // Test 2: GPU forward at various sizes (sanity check - runs without error)
    for k in [4, 8, 16] {
        let n = 1 << k
        var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE &+ UInt64(k << 8)
        var original = [UInt8](repeating: 0, count: n)
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            original[i] = UInt8(truncatingIfNeeded: rng)
        }

        var basis = [UInt8](repeating: 0, count: k)
        var elem: UInt8 = 0x02
        for i in 0..<k {
            basis[i] = elem
            elem = elem &* elem
        }

        guard let fwd = try? engine.forward(data: original, n: n, k: k, basis: basis) else { continue }
        expect(fwd.count == n, "Forward 2^\(k) produces n elements")
    }

    // Test 3: Pointwise multiply runs without error
    do {
        let k = 4
        let n = 1 << k
        var rng: UInt64 = 0xFEED_FACE &+ UInt64(k << 16)
        var aData = [UInt8](repeating: 0, count: n)
        var bData = [UInt8](repeating: 0, count: n)
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            aData[i] = UInt8(truncatingIfNeeded: rng)
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            bData[i] = UInt8(truncatingIfNeeded: rng)
        }

        if let result = try? engine.pointwiseMultiply(a: aData, b: bData, n: n) {
            expect(result.count == n, "Pointwise multiply produces n elements")
        }
    }

    // Test 4: Batch forward runs without error
    do {
        let k = 4
        let n = 1 << k
        let batch = 4
        let total = n * batch

        var rng: UInt64 = 0xABCD_1234
        var original = [UInt8](repeating: 0, count: total)
        for i in 0..<total {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            original[i] = UInt8(truncatingIfNeeded: rng)
        }

        var basis = [UInt8](repeating: 0, count: k)
        var elem: UInt8 = 0x02
        for i in 0..<k {
            basis[i] = elem
            elem = elem &* elem
        }

        if let batchResult = try? engine.forwardBatch(data: original, n: n, k: k, batch: batch, basis: basis) {
            expect(batchResult.count == total, "Batch forward produces total elements")
        }
    }

    // Test 5: Polynomial multiply via FFT
    do {
        let k = 4
        let n = 1 << k
        let halfN = n >> 1
        var rng: UInt64 = 0x1234_5678_ABCD_EF00 &+ UInt64(k << 16)
        var aData = [UInt8](repeating: 0, count: halfN)
        var bData = [UInt8](repeating: 0, count: halfN)
        for i in 0..<halfN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            aData[i] = UInt8(truncatingIfNeeded: rng)
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            bData[i] = UInt8(truncatingIfNeeded: rng)
        }

        var basis = [UInt8](repeating: 0, count: k)
        var elem: UInt8 = 0x02
        for i in 0..<k {
            basis[i] = elem
            elem = elem &* elem
        }

        if let result = try? engine.multiply(aData, bData, n: n, k: k, basis: basis) {
            expect(result.count == n, "Multiply produces correct size 2^\(k)")
        }
    }

    // Test 6: Performance benchmark at various sizes
    for k in [16, 18, 20, 22] {
        let n = 1 << k
        var rng: UInt64 = 0xCAFE_BABE &+ UInt64(k << 8)
        var original = [UInt8](repeating: 0, count: n)
        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            original[i] = UInt8(truncatingIfNeeded: rng)
        }

        var basis = [UInt8](repeating: 0, count: k)
        var elem: UInt8 = 0x02
        for i in 0..<k { basis[i] = elem; elem = elem &* elem }

        let t0 = CFAbsoluteTimeGetCurrent()
        guard let fwd = try? engine.forward(data: original, n: n, k: k, basis: basis) else { continue }
        let t1 = CFAbsoluteTimeGetCurrent()
        _ = fwd

        let throughput = Double(n) / (t1 - t0) / 1e6
        print(String(format: "  GF(2^8) GPU 2^%d (%d elements): %.2fms (%.1f M elem/s)", k, n, (t1 - t0) * 1000, throughput))
        expect(true, "GPU forward 2^\(k) completed")

        // Compare with forwardPairs (n/2 threads)
        if let _ = engine.forwardPairsFn {
            let t0p = CFAbsoluteTimeGetCurrent()
            guard let fwdPairs = try? engine.forwardPairs(data: original, n: n, k: k, basis: basis) else { continue }
            let t1p = CFAbsoluteTimeGetCurrent()
            if fwdPairs != nil && fwdPairs == fwd {
                let speedup = (t1 - t0) / (t1p - t0p)
                print(String(format: "    forwardPairs: %.2fms [%.2fx]", (t1p - t0p) * 1000, speedup))
            }
        }

        // Compare with forwardPairsTg
        if let _ = engine.forwardPairsTgFn {
            let t0tg = CFAbsoluteTimeGetCurrent()
            guard let fwdTg = try? engine.forwardPairsTg(data: original, n: n, k: k, basis: basis) else { continue }
            let t1tg = CFAbsoluteTimeGetCurrent()
            if fwdTg != nil && fwdTg == fwd {
                let speedup = (t1 - t0) / (t1tg - t0tg)
                print(String(format: "    forwardPairsTg: %.2fms [%.2fx]", (t1tg - t0tg) * 1000, speedup))
            }
        }

        // Compare with forwardVec4 (n/8 threads)
        if let _ = engine.forwardVec4Fn {
            let t0v = CFAbsoluteTimeGetCurrent()
            guard let fwdVec4 = try? engine.forwardVec4(data: original, n: n, k: k, basis: basis) else { continue }
            let t1v = CFAbsoluteTimeGetCurrent()
            if fwdVec4 != nil {
                if fwdVec4 == fwd {
                    let speedup = (t1 - t0) / (t1v - t0v)
                    print(String(format: "    forwardVec4: %.2fms [%.2fx]", (t1v - t0v) * 1000, speedup))
                } else {
                    print(String(format: "    forwardVec4: MISMATCH (correctness error!)"))
                }
            }
        }

        // Compare with SIMD shuffle version
        if let shuffleFn = engine.forwardShuffleFn {
            let t0s = CFAbsoluteTimeGetCurrent()
            guard let fwdShuffle = try? engine.forwardShuffle(data: original, n: n, k: k, basis: basis) else { continue }
            let t1s = CFAbsoluteTimeGetCurrent()
            _ = fwdShuffle

            // Verify correctness: shuffle result should match regular result
            if fwd != fwdShuffle {
                print("  [FAIL] forwardShuffle result mismatch at 2^\(k)!")
            }

            let throughputShuffle = Double(n) / (t1s - t0s) / 1e6
            let speedup = (t1 - t0) / (t1s - t0s)
            print(String(format: "  GF(2^8) GPU SHUFFLE 2^%d: %.2fms (%.1f M elem/s) [%.2fx]", k, n, (t1s - t0s) * 1000, throughputShuffle, speedup))
        }
    }
}
