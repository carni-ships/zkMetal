// ZKMetalDemo — Minimal demo of zkMetal GPU-accelerated ZK proving
//
// Demonstrates:
// 1. BN254 MSM (multi-scalar multiplication) on Metal GPU
// 2. Poseidon2 hash on Metal GPU
// 3. Groth16 proof generation and verification

import Foundation
import Metal
import zkMetal

// MARK: - Helpers

func timeIt<T>(_ label: String, _ body: () throws -> T) rethrows -> T {
    let t0 = CFAbsoluteTimeGetCurrent()
    let result = try body()
    let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
    print(String(format: "  %-40s %8.2f ms", label, ms))
    return result
}

// MARK: - MSM Benchmark

func runMSMBench() throws {
    print("\n=== BN254 MSM Benchmark ===")

    let engine = try MetalMSM()
    print("  GPU: \(engine.device.name)")

    let logN = 14
    let n = 1 << logN

    // Generate n distinct points on BN254 G1
    let g = pointFromAffine(bn254G1Generator())
    var projPoints = [PointProjective]()
    projPoints.reserveCapacity(n)
    var acc = g
    for _ in 0..<n {
        projPoints.append(acc)
        acc = pointAdd(acc, g)
    }
    let points = batchToAffine(projPoints)
    projPoints = [] // free memory

    // Generate random scalars
    var rng: UInt64 = 0xDEAD_BEEF_CAFE_BABE
    var scalars = [[UInt32]]()
    scalars.reserveCapacity(n)
    for _ in 0..<n {
        var limbs = [UInt32](repeating: 0, count: 8)
        for j in 0..<8 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            limbs[j] = UInt32(truncatingIfNeeded: rng >> 32)
        }
        scalars.append(limbs)
    }

    // Warmup
    let _ = try engine.msm(points: points, scalars: scalars)

    // Benchmark (3 iterations, report best)
    var bestMs = Double.infinity
    for _ in 0..<3 {
        let t0 = CFAbsoluteTimeGetCurrent()
        let _ = try engine.msm(points: points, scalars: scalars)
        let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        bestMs = min(bestMs, ms)
    }
    print(String(format: "  MSM 2^%d (%d points):                    %8.2f ms (best of 3)", logN, n, bestMs))
}

// MARK: - Poseidon2 Hash Benchmark

func runPoseidon2Bench() throws {
    print("\n=== Poseidon2 Hash Benchmark ===")

    let engine = try Poseidon2Engine()

    let numPairs = 1024
    var input = [Fr]()
    input.reserveCapacity(numPairs * 2)
    for i in 0..<(numPairs * 2) {
        input.append(frFromInt(UInt64(i + 1)))
    }

    // Warmup
    let _ = try engine.hashPairs(input)

    // Benchmark
    let hashes = try timeIt("Poseidon2 hash \(numPairs) pairs (GPU)") {
        try engine.hashPairs(input)
    }

    // Verify output is non-trivial
    let first = hashes[0]
    let isNonZero = !first.isZero
    print("  Output[0] non-zero: \(isNonZero)")
    print("  Hashes computed: \(hashes.count)")
}

// MARK: - Groth16 Prove + Verify

func runGroth16Demo() throws {
    print("\n=== Groth16 Prove & Verify ===")
    print("  Circuit: x^3 + x + 5 = y")

    // Build circuit and witness for x = 3 => y = 35
    let r1cs = buildExampleCircuit()
    let (publicInputs, witness) = computeExampleWitness(x: 3)
    print("  Public inputs: x=3, y=35")
    print("  Constraints: \(r1cs.numConstraints), Variables: \(r1cs.numVars)")

    // Trusted setup
    let setup = Groth16Setup()
    let (pk, vk) = timeIt("Trusted setup") {
        setup.setup(r1cs: r1cs)
    }

    // Prove
    let prover = try Groth16Prover()
    let proof = try timeIt("Prove") {
        try prover.prove(pk: pk, r1cs: r1cs, publicInputs: publicInputs, witness: witness)
    }

    // Verify
    let verifier = Groth16Verifier()
    let valid = timeIt("Verify") {
        verifier.verify(proof: proof, vk: vk, publicInputs: publicInputs)
    }
    print("  Proof valid: \(valid)")

    // Verify with wrong input should fail
    let wrongInputs = [frFromInt(3), frFromInt(99)]
    let invalid = verifier.verify(proof: proof, vk: vk, publicInputs: wrongInputs)
    print("  Reject bad input: \(!invalid)")
}

// MARK: - Main

print("zkMetal Demo v\(ZKMetal.version)")
print("============================================")

guard MTLCreateSystemDefaultDevice() != nil else {
    print("ERROR: No Metal GPU available.")
    print("This demo requires Apple Silicon (M1+) or A-series GPU.")
    exit(1)
}

do {
    try runMSMBench()
    try runPoseidon2Bench()
    try runGroth16Demo()

    print("\n============================================")
    print("All demos completed successfully.")
} catch {
    print("\nERROR: \(error)")
    exit(1)
}
