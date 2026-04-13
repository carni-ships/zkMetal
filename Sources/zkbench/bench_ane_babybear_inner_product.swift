// bench_ane_babybear_inner_product.swift — ANE BabyBear Inner Product Benchmark
//
// Tests ANEBabyBearInnerProductEngine with CPU and GPU paths.

import Foundation
import Metal
import zkMetal

public func runANEBabyBearInnerProductBench() {
    fputs("ANE BabyBear Inner Product Benchmark\n", stderr)

    guard let device = MTLCreateSystemDefaultDevice() else {
        fputs("Error: No Metal GPU available\n", stderr)
        return
    }
    fputs("Device: \(device.name)\n", stderr)

    // Initialize ANE tensor subsystem
    let aneAvailable = ANEBabyBearInnerProductEngine.initializeANE()
    fputs("ANE/GPU available: \(aneAvailable)\n", stderr)

    let engine = ANEBabyBearInnerProductEngine()

    // Test with various vector sizes
    let sizes = [8, 16, 32, 64, 128, 256, 512, 1024]

    // Generate test vectors
    let seed: UInt64 = 0xDEAD_BEEF_CAFE_BABE
    var rng = seed

    for n in sizes {
        var a = [UInt32](repeating: 0, count: n)
        var b = [UInt32](repeating: 0, count: n)

        for i in 0..<n {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            a[i] = UInt32(truncatingIfNeeded: rng >> 32) % 0x78000001

            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            b[i] = UInt32(truncatingIfNeeded: rng >> 32) % 0x78000001
        }

        // Benchmark CPU path (forceANE = false, high threshold)
        engine.forceANE = false
        engine.aneThreshold = 1024

        let cpuT0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<100 {
            _ = engine.innerProduct(a, b)
        }
        let cpuT1 = CFAbsoluteTimeGetCurrent()
        let cpuTime = (cpuT1 - cpuT0) * 1000 / 100.0

        fputs(String(format: "  n=%4d: CPU %.3f ms/iter\n", n, cpuTime), stderr)
    }

    fputs("Done.\n", stderr)
}