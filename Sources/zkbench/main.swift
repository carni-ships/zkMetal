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
    let logSizes = CommandLine.arguments.contains("--quick") ? [16, 18] : [8, 10, 12, 14, 16, 17, 18, 20]
    let sizes = logSizes.map { 1 << $0 }
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

    let doProfile = CommandLine.arguments.contains("--profile")

    for n in sizes {
        let points = Array(allPoints.prefix(n))
        let scalars = Array(allScalars.prefix(n))

        // Warmup
        let _ = try engine.msm(points: points, scalars: scalars)

        // Profile run (one extra run with instrumentation)
        if doProfile && (n == 1 << 16 || n == 1 << 18) {
            fputs("  --- Profile for 2^\(Int(log2(Double(n)))) ---\n", stderr)
            engine.profileMSM = true
            let _ = try engine.msm(points: points, scalars: scalars)
            engine.profileMSM = false
        }

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
        let logN = logSizes[sizes.firstIndex(of: n)!]

        // CPU baseline (sequential scalar mul + accumulate) — skip for large sizes
        if !skipCPU && n <= 16384 {
            let projPoints = points.map { pointFromAffine($0) }
            let cpuT0 = CFAbsoluteTimeGetCurrent()
            var cpuResult = pointIdentity()
            for i in 0..<n {
                let scalar = frFromLimbs(scalars[i])
                cpuResult = pointAdd(cpuResult, pointScalarMul(projPoints[i], scalar))
            }
            _ = cpuResult  // prevent optimization
            let cpuTime = (CFAbsoluteTimeGetCurrent() - cpuT0) * 1000
            let speedup = cpuTime / median
            fputs(String(format: "  MSM 2^%-2d = %7d pts: GPU %7.1f ms | CPU %8.1f ms | %.1f×\n",
                        logN, n, median, cpuTime, speedup), stderr)
        } else {
            fputs(String(format: "  MSM 2^%-2d = %7d pts: %7.1f ms\n", logN, n, median), stderr)
        }
    }
}

/// Global flag: skip CPU benchmarks when --no-cpu is passed
public var skipCPU = CommandLine.arguments.contains("--no-cpu")

let args = CommandLine.arguments.filter { !$0.hasPrefix("--") }
let cmd = args.count >= 2 ? args[1] : "msm"
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
} else if cmd == "poseidon2-m31" || cmd == "p2m31" {
    runPoseidon2M31Bench()
} else if cmd == "keccak" || cmd == "k256" {
    runKeccakBench()
} else if cmd == "merkle" {
    runMerkleBench()
} else if cmd == "imerkle" || cmd == "incremental-merkle" {
    runIncrementalMerkleBench()
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
} else if cmd == "kzg-batch" {
    runKZGBatchBench()
} else if cmd == "bls377" || cmd == "bls12377" {
    runBLS12377NTTBench()
} else if cmd == "bls377msm" || cmd == "bls12377msm" {
    runBLS12377MSMBench()
} else if cmd == "bls377glv" {
    runBLS12377GLVTest()
} else if cmd == "bls381" || cmd == "bls12381" {
    runBLS12381Test()
} else if cmd == "blake3" || cmd == "b3" {
    runBlake3Bench()
} else if cmd == "secp256k1" || cmd == "secp" {
    runSecp256k1Test()
} else if cmd == "secpglv" {
    runSecp256k1GLVTest()
} else if cmd == "secpmsm" {
    runSecp256k1MSMBench()
} else if cmd == "ecdsa" {
    runECDSABench()
} else if cmd == "ipa" {
    runIPABench()
} else if cmd == "verkle" {
    runVerkleBench()
} else if cmd == "lookup" || cmd == "logup" {
    runLookupBench()
} else if cmd == "lasso" {
    runLassoBench()
// } else if cmd == "cq" || cmd == "cached-quotients" {
//     runCQBench()
} else if cmd == "sparse" || cmd == "sparse-sumcheck" {
    runSparseSumcheckBench()
} else if cmd == "batch-field" || cmd == "bf" {
    runBatchFieldBench()
} else if cmd == "witness" || cmd == "trace" {
    runWitnessBench()
} else if cmd == "constraint" || cmd == "ir" {
    runConstraintBench()
} else if cmd == "fused" || cmd == "fused-ntt" {
    runFusedConstraintBench()
} else if cmd == "circle" || cmd == "m31" {
    runCircleBench()
} else if cmd == "circle-stark" || cmd == "cstark" {
    runCircleSTARKBench()
} else if cmd == "circle-fri" || cmd == "cfri" {
    runCircleFRIBench()
} else if cmd == "transcript" || cmd == "ts" {
    runTranscriptBench()
} else if cmd == "serial" || cmd == "serialize" {
    runSerializationBench()
} else if cmd == "basefold" || cmd == "bf-pcs" {
    runBasefoldBench()
} else if cmd == "brakedown" || cmd == "bk" {
    runBrakedownBench()
} else if cmd == "pasta" {
    runPastaTest()
} else if cmd == "pastamsm" || cmd == "pasta-msm" {
    runPastaMSMBench()
} else if cmd == "binius" || cmd == "binary" || cmd == "bt" {
    runBiniusBench()
} else if cmd == "fold" || cmd == "nova" || cmd == "hypernova" {
    runFoldingBench()
} else if cmd == "gkr" {
    runGKRBench()
} else if cmd == "batch-verify" || cmd == "bv" {
    runBatchVerifyBench()
} else if cmd == "sort" || cmd == "radix" {
    runSortBench()
} else if cmd == "test-parallel-ntt" {
    runParallelNTTTests()
} else if cmd == "versions" || cmd == "version" {
    Versions.printAll()
} else if cmd == "test" {
    runAllCorrectnessTests()
} else if cmd == "asm" || cmd == "asm-mont" {
    runAsmMontBench()
} else if cmd == "gl-neon" {
    runGoldilocksNeonBench()
} else if cmd == "keccak-neon" || cmd == "kn" {
    runKeccakNeonBench()
} else if cmd == "blake3-neon" || cmd == "b3-neon" {
    runBlake3NeonBench()
} else if cmd == "cpu" {
    runCPUBench()
} else if cmd == "cpu-msm" {
    runCPUMSMBench()
} else if cmd == "msm" {
    try runMSMBench()
} else if cmd == "all" {
    try runMSMBench()
    runNTTBench()
    runBLS12377NTTBench()
    runPoseidon2Bench()
    runKeccakBench()
    runMerkleBench()
    runPolyBench()
    runFRIBench()
    runSumcheckBench()
    runKZGBench()
    runBlake3Bench()
    runIPABench()
    runVerkleBench()
    runLookupBench()
    runLassoBench()
    runSparseSumcheckBench()
    runSortBench()
    runECDSABench()
    runCircleBench()
    runCircleFRIBench()
    runTranscriptBench()
    runBasefoldBench()
    runGKRBench()
} else {
    try runMSMBench()
}
