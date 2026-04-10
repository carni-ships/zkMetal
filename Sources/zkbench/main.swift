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
    let paperMode = CommandLine.arguments.contains("--paper")
    let iterations = paperMode ? 20 : 10

    var msmResults = [(String, BenchResult)]()

    for n in sizes {
        let points = Array(allPoints.prefix(n))
        let scalars = Array(allScalars.prefix(n))
        let logN = logSizes[sizes.firstIndex(of: n)!]

        // Profile run (one extra run with instrumentation)
        if doProfile && (n == 1 << 16 || n == 1 << 18) {
            fputs("  --- Profile for 2^\(logN) ---\n", stderr)
            engine.profileMSM = true
            let _ = try engine.msm(points: points, scalars: scalars)
            engine.profileMSM = false
        }

        // GPU benchmark with full statistics
        let label = String(format: "MSM 2^%-2d = %7d pts", logN, n)
        let gpuResult = try bench(label, warmup: 2, iterations: iterations) {
            let _ = try engine.msm(points: points, scalars: scalars)
        }
        msmResults.append((label, gpuResult))

        // CPU baseline (sequential scalar mul + accumulate) — skip for large sizes
        if !skipCPU && n <= 16384 {
            let cpuLabel = String(format: "MSM 2^%-2d CPU", logN)
            let cpuResult = bench(cpuLabel, warmup: 0, iterations: 3) {
                let projPoints = points.map { pointFromAffine($0) }
                var cpuR = pointIdentity()
                for i in 0..<n {
                    let scalar = frFromLimbs(scalars[i])
                    cpuR = pointAdd(cpuR, pointScalarMul(projPoints[i], scalar))
                }
                _ = cpuR
            }
            let speedup = cpuResult.median / gpuResult.median
            fputs(String(format: "    -> GPU vs CPU: %.1fx speedup\n", speedup), stderr)
        }
    }

    // Paper output (--paper flag)
    if paperMode {
        fputs("\n--- LaTeX Table ---\n", stderr)
        print(formatLatexTable(results: msmResults))
        fputs("\n--- Markdown Table ---\n", stderr)
        print(formatMarkdownTable(results: msmResults))
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
} else if cmd == "poseidon2-bb" || cmd == "p2bb" {
    runPoseidon2BabyBearBench()
} else if cmd == "keccak" || cmd == "k256" {
    runKeccakBench()
} else if cmd == "merkle" {
    runMerkleBench()
} else if cmd == "merkle-tree" || cmd == "mt" || cmd == "gpu-merkle" {
    runMerkleTreeBench()
} else if cmd == "imerkle" || cmd == "incremental-merkle" {
    runIncrementalMerkleBench()
} else if cmd == "poly" {
    runPolyBench()
} else if cmd == "fields" {
    runFieldBench()
} else if cmd == "fri" {
    runFRIBench()
} else if cmd == "stir" {
    runSTIRBench()
} else if cmd == "whir" {
    runWHIRBench()
} else if cmd == "sumcheck" || cmd == "sc" {
    runSumcheckBench()
} else if cmd == "hdsumcheck" || cmd == "hdsc" {
    runHighDegSumcheckBench()
} else if cmd == "usumcheck" || cmd == "usc" {
    runUnivariateSumcheckBench()
} else if cmd == "kzg" {
    runKZGBench()
} else if cmd == "kzg-batch" {
    runKZGBatchBench()
} else if cmd == "kzg-fused" {
    runKZGFusedBench()
} else if cmd == "gpu-kzg" || cmd == "gpukzg" {
    runGPUKZGBench()
} else if cmd == "bls377" || cmd == "bls12377" {
    runBLS12377NTTBench()
} else if cmd == "bls377msm" || cmd == "bls12377msm" {
    runBLS12377MSMBench()
} else if cmd == "bls377glv" {
    runBLS12377GLVTest()
} else if cmd == "stark252" || cmd == "starknet" {
    runStark252Bench()
} else if cmd == "bls381msm" {
    runBLS12381MSMBench()
} else if cmd == "bls12381" {
    runBLS12381Test()
} else if cmd == "sha256" || cmd == "sha" {
    runBlake3Bench()
} else if cmd == "secp256k1" || cmd == "secp" {
    runSecp256k1Test()
} else if cmd == "secpglv" {
    runSecp256k1GLVTest()
} else if cmd == "secpmsm" {
    runSecp256k1MSMBench()
} else if cmd == "ecdsa" {
    runECDSABench()
} else if cmd == "batch-ecdsa" || cmd == "batchecdsa" {
    runBatchECDSABench()
} else if cmd == "schnorr" || cmd == "bip340" || cmd == "taproot" {
    runSchnorrBench()
} else if cmd == "bls-sig" || cmd == "bls-signature" || cmd == "blssig" {
    runBLSSignatureBench()
} else if cmd == "ed25519" || cmd == "eddsa" || cmd == "curve25519" {
    runEd25519Bench()
} else if cmd == "batch-ed25519" || cmd == "batched25519" {
    runBatchEd25519Bench()
} else if cmd == "babyjubjub" || cmd == "bjj" || cmd == "pedersen-bjj" {
    runBabyJubjubBench()
} else if cmd == "jubjub" {
    runJubjubBench()
} else if cmd == "ipa" {
    runIPABench()
} else if cmd == "verkle" {
    runVerkleBench()
} else if cmd == "lookup" || cmd == "logup" {
    runLookupBench()
} else if cmd == "plookup" {
    runPlookupBench()
} else if cmd == "lasso" {
    runLassoBench()
} else if cmd == "cq" || cmd == "cached-quotients" {
    runCQLookupBench()
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
} else if cmd == "fused-general" || cmd == "fg" {
    runFusedGeneralConstraintBench()
} else if cmd == "fused-circle" || cmd == "fc" {
    runFusedCircleConstraintBench()
} else if cmd == "circle" || cmd == "m31" {
    runCircleBench()
} else if cmd == "circle-stark" || cmd == "cstark" {
    runCircleSTARKBench()
} else if cmd == "circle-stark-fused" || cmd == "cstark-fused" {
    runCircleSTARKFusedRoundBench()
} else if cmd == "circle-fri" || cmd == "cfri" {
    runCircleFRIBench()
} else if cmd == "transcript" || cmd == "ts" {
    runTranscriptBench()
} else if cmd == "batch-transcript" || cmd == "bts" {
    runBatchTranscriptBench()
} else if cmd == "serial" || cmd == "serialize" {
    runSerializationBench()
} else if cmd == "pcs" || cmd == "pcs-compare" || cmd == "pcs-comparison" {
    runPCSComparisonBench()
} else if cmd == "basefold" || cmd == "bf-pcs" {
    runBasefoldBench()
} else if cmd == "zeromorph" || cmd == "zm" {
    runZeromorphBench()
} else if cmd == "brakedown" || cmd == "bk" {
    runBrakedownBench()
} else if cmd == "grumpkin" || cmd == "grump" {
    runGrumpkinTest()
} else if cmd == "grumpkinmsm" || cmd == "grumpkin-msm" {
    runGrumpkinMSMBench()
} else if cmd == "g2msm" || cmd == "g2-msm" || cmd == "bn254g2" {
    runG2MSMBench()
} else if cmd == "pasta" {
    runPastaTest()
} else if cmd == "pastamsm" || cmd == "pasta-msm" {
    runPastaMSMBench()
} else if cmd == "binius" || cmd == "binary" || cmd == "bt" {
    runBiniusBench()
} else if cmd == "fold" || cmd == "nova" || cmd == "hypernova" {
    runFoldingBench()
} else if cmd == "accum" || cmd == "accumulate" || cmd == "accumulation" {
    runAccumulationBench()
} else if cmd == "gkr" {
    runGKRBench()
} else if cmd == "dparallel" || cmd == "dp" || cmd == "datapar" {
    runDataParallelBench()
} else if cmd == "batch-verify" || cmd == "bv" {
    runBatchVerifyBench()
} else if cmd == "stream-verify" || cmd == "streaming-verify" || cmd == "sv" {
    runStreamingVerifyBench()
} else if cmd == "lattice" || cmd == "kyber" || cmd == "dilithium" || cmd == "pq" {
    runLatticeBench()
} else if cmd == "rs" || cmd == "erasure" || cmd == "reed-solomon" {
    runErasureBench()
} else if cmd == "sort" || cmd == "radix" {
    runSortBench()
} else if cmd == "test-parallel-ntt" {
    runParallelNTTTests()
} else if cmd == "versions" || cmd == "version" {
    Versions.printAll()
} else if cmd == "test" {
    runAllCorrectnessTests()
} else if cmd == "field-ops" || cmd == "fieldops" || cmd == "fo" {
    runFieldOpsBench()
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
} else if cmd == "plonk" {
    runPlonkBench()
} else if cmd == "plonk-custom" || cmd == "plonk_custom" {
    runPlonkCustomBench()
} else if cmd == "marlin" || cmd == "ahp" {
    runMarlinBench()
} else if cmd == "tensor" || cmd == "tc" {
    runTensorBench()
} else if cmd == "spartan" {
    runSpartanBench()
} else if cmd == "pairing" || cmd == "pair" {
    runPairingBench()
} else if cmd == "groth16" || cmd == "g16" {
    runGroth16Bench()
} else if cmd == "jolt" || cmd == "vm" || cmd == "zkvm" {
    runJoltBench()
} else if cmd == "msm" {
    try runMSMBench()
} else if cmd == "all" {
    try runMSMBench()
    runNTTBench()
    runBLS12377NTTBench()
    runStark252Bench()
    runPoseidon2Bench()
    runPoseidon2BabyBearBench()
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
    runPlookupBench()
    runLassoBench()
    runSparseSumcheckBench()
    runHighDegSumcheckBench()
    runSortBench()
    runECDSABench()
    runBatchECDSABench()
    runCircleBench()
    runCircleFRIBench()
    runTranscriptBench()
    runBatchTranscriptBench()
    runBasefoldBench()
    runGKRBench()
    runPlonkBench()
    runGroth16Bench()
    // runMarlinBench()
    // fputs("stub\n", stderr)
    runErasureBench()
} else {
    try runMSMBench()
}
