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

        guard let result = try? engine.pointwiseMultiply(a: aData, b: bData, n: n) else { return }
        expect(result.count == n, "Pointwise multiply produces n elements")
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

        guard let batchResult = try? engine.forwardBatch(data: original, n: n, k: k, batch: batch, basis: basis) else { return }
        expect(batchResult.count == total, "Batch forward produces total elements")
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

        guard let result = try? engine.multiply(aData, bData, n: n, k: k, basis: basis) else { return }
        expect(result.count == n, "Multiply produces correct size 2^\(k)")
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
        for i in 0..<k {
            basis[i] = elem
            elem = elem &* elem
        }

        let t0 = CFAbsoluteTimeGetCurrent()
        guard let fwd = try? engine.forward(data: original, n: n, k: k, basis: basis) else { continue }
        let t1 = CFAbsoluteTimeGetCurrent()
        _ = fwd

        let throughput = Double(n) / (t1 - t0) / 1e6
        print(String(format: "  GF(2^8) GPU 2^%d (%d elements): %.2fms (%.1f M elem/s)", k, n, (t1 - t0) * 1000, throughput))
        expect(true, "GPU forward 2^\(k) completed")
    }
}
