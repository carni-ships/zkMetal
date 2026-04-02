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

    let args = CommandLine.arguments
    let n = args.count >= 3 ? (Int(args[2]) ?? 65536) : 65536

    // MSM benchmark
    fputs("\n--- MSM Benchmark ---\n", stderr)
    let engine = try MetalMSM()

    let gx = fpFromInt(1)
    let gy = fpFromInt(2)
    let g = PointAffine(x: gx, y: gy)

    fputs("Generating \(n) distinct points...\n", stderr)
    var projPoints = [PointProjective]()
    projPoints.reserveCapacity(n)
    let gProj = pointFromAffine(g)
    var acc = gProj
    for _ in 0..<n {
        projPoints.append(acc)
        acc = pointAdd(acc, gProj)
    }
    let points = batchToAffine(projPoints)

    var scalars: [[UInt32]] = []
    var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
    for _ in 0..<n {
        var limbs: [UInt32] = Array(repeating: 0, count: 8)
        for j in 0..<8 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            limbs[j] = UInt32(truncatingIfNeeded: rng >> 32)
        }
        scalars.append(limbs)
    }

    // Warmup
    let _ = try engine.msm(points: points, scalars: scalars)

    // Timed runs
    let runs = 5
    var times = [Double]()
    for i in 0..<runs {
        let start = CFAbsoluteTimeGetCurrent()
        let _ = try engine.msm(points: points, scalars: scalars)
        let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
        times.append(elapsed)
        fputs("  run \(i+1): \(String(format: "%.1f", elapsed))ms\n", stderr)
    }
    times.sort()
    let median = times[runs / 2]
    let best = times[0]
    fputs("MSM(\(n)): median \(String(format: "%.1f", median))ms, best \(String(format: "%.1f", best))ms\n", stderr)
}

let cmd = CommandLine.arguments.count >= 2 ? CommandLine.arguments[1] : "msm"
if cmd == "ntt" {
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
} else if cmd == "sort" || cmd == "radix" {
    runSortBench()
} else if cmd == "fri" {
    runFRIBench()
} else if cmd == "sumcheck" || cmd == "sc" {
    runSumcheckBench()
} else if cmd == "all" {
    runNTTBench()
    runPoseidon2Bench()
    runKeccakBench()
    runMerkleBench()
    runPolyBench()
    runSortBench()
    runFRIBench()
    runSumcheckBench()
} else {
    try runMSMBench()
}
