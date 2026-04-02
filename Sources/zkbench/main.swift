// zkbench — Benchmark harness for zkMetal primitives

import Foundation
import Metal
import zkMetal

func runMSMBench() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
        fputs("Error: No Metal GPU available\n", stderr)
        return
    }
    fputs("zkbench — \(device.name)\n", stderr)

    fputs("\n--- MSM Benchmark (BN254 G1) ---\n", stderr)
    let engine = try MetalMSM()

    let gx = fpFromInt(1)
    let gy = fpFromInt(2)
    let gProj = pointFromAffine(PointAffine(x: gx, y: gy))

    // Generate max points once, slice for smaller sizes
    let sizes = [256, 1024, 4096, 16384, 65536, 131072, 262144]
    let maxN = sizes.last!

    fputs("Generating \(maxN) distinct points...\n", stderr)
    let genT0 = CFAbsoluteTimeGetCurrent()
    var projPoints = [PointProjective]()
    projPoints.reserveCapacity(maxN)
    var acc = gProj
    for _ in 0..<maxN {
        projPoints.append(acc)
        acc = pointAdd(acc, gProj)
    }
    let allPoints = batchToAffine(projPoints)
    projPoints = []  // free memory
    fputs("Point generation: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent() - genT0) * 1000))ms\n", stderr)

    var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
    var allScalars = [[UInt32]]()
    allScalars.reserveCapacity(maxN)
    for _ in 0..<maxN {
        var limbs = [UInt32](repeating: 0, count: 8)
        for j in 0..<8 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            limbs[j] = UInt32(truncatingIfNeeded: rng >> 32)
        }
        allScalars.append(limbs)
    }

    for n in sizes {
        let points = Array(allPoints.prefix(n))
        let scalars = Array(allScalars.prefix(n))

        // Warmup
        let _ = try engine.msm(points: points, scalars: scalars)

        // Timed runs
        let runs = 5
        var times = [Double]()
        for _ in 0..<runs {
            let start = CFAbsoluteTimeGetCurrent()
            let _ = try engine.msm(points: points, scalars: scalars)
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            times.append(elapsed)
        }
        times.sort()
        let median = times[runs / 2]
        fputs(String(format: "  MSM %7d pts: %7.1f ms\n", n, median), stderr)
    }
}

let cmd = CommandLine.arguments.count >= 2 ? CommandLine.arguments[1] : "msm"
if cmd == "calibrate" {
    guard let device = MTLCreateSystemDefaultDevice() else {
        fputs("Error: No Metal GPU available\n", stderr)
        exit(1)
    }
    let config = TuningManager.shared.recalibrate(device: device)
    print("Tuning config for \(config.deviceName):")
    print("  NTT threadgroup size:      \(config.nttThreadgroupSize)")
    print("  NTT four-step threshold:   \(config.nttFourStepThreshold)")
    print("  MSM threadgroup size:      \(config.msmThreadgroupSize)")
    print("  MSM window bits (large):   \(config.msmWindowBitsLarge)")
    print("  Hash threadgroup size:     \(config.hashThreadgroupSize)")
    print("  FRI threadgroup size:      \(config.friThreadgroupSize)")
    print("  Sumcheck fused TG size:    \(config.sumcheckFusedTGSize)")
    print("  Sumcheck per-round TG:     \(config.sumcheckPerRoundTGSize)")
    print("\nSaved to ~/.zkmetal/tuning.json")
} else if cmd == "ntt" {
    runNTTBench()
} else if cmd == "poseidon2" || cmd == "p2" {
    runPoseidon2Bench()
} else if cmd == "keccak" || cmd == "k256" {
    runKeccakBench()
} else if cmd == "merkle" {
    runMerkleBench()
} else if cmd == "poly" {
    runPolyBench()
} else if cmd == "fields" {
    runFieldBench()
} else if cmd == "fri" {
    runFRIBench()
} else if cmd == "sumcheck" || cmd == "sc" {
    runSumcheckBench()
} else if cmd == "kzg" {
    runKZGBench()
} else if cmd == "all" {
    runNTTBench()
    runPoseidon2Bench()
    runKeccakBench()
    runMerkleBench()
    runPolyBench()
    runFRIBench()
    runSumcheckBench()
    runKZGBench()
} else {
    try runMSMBench()
}
