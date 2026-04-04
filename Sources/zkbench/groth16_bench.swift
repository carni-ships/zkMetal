// Groth16 SNARK Benchmark
// Tests: example circuit (x^3 + x + 5 = y), larger circuits, prove + verify

import Foundation
import Metal
import zkMetal

func runGroth16Bench() {
    fputs("\n--- Groth16 SNARK Benchmark (BN254) ---\n", stderr)

    // 1. Example circuit: x^3 + x + 5 = y
    fputs("\n[1] Example circuit: x^3 + x + 5 = y\n", stderr)
    let r1cs = buildExampleCircuit()
    let (pubInputs, witness) = computeExampleWitness(x: 3)

    // Verify R1CS satisfaction
    var z = [Fr](repeating: Fr.zero, count: r1cs.numVars)
    z[0] = Fr.one
    z[1] = pubInputs[0]  // x
    z[2] = pubInputs[1]  // y
    for i in 0..<witness.count {
        z[3 + i] = witness[i]
    }
    let satisfied = r1cs.isSatisfied(z: z)
    fputs("  R1CS satisfied: \(satisfied)\n", stderr)

    if !satisfied {
        fputs("  ERROR: R1CS not satisfied, aborting\n", stderr)
        return
    }

    // Trusted setup
    let setup = Groth16Setup()
    let setupT0 = CFAbsoluteTimeGetCurrent()
    let (pk, vk) = setup.setup(r1cs: r1cs)
    let setupTime = (CFAbsoluteTimeGetCurrent() - setupT0) * 1000
    fputs("  Setup: \(String(format: "%.1f", setupTime))ms\n", stderr)
    fputs("  PK sizes: a_query=\(pk.a_query.count), b_g2_query=\(pk.b_g2_query.count), h_query=\(pk.h_query.count), l_query=\(pk.l_query.count)\n", stderr)

    // Prove
    do {
        let prover = try Groth16Prover()

        let proveT0 = CFAbsoluteTimeGetCurrent()
        let proof = try prover.prove(pk: pk, r1cs: r1cs,
                                      publicInputs: pubInputs, witness: witness)
        let proveTime = (CFAbsoluteTimeGetCurrent() - proveT0) * 1000
        fputs("  Prove: \(String(format: "%.1f", proveTime))ms\n", stderr)

        // Show proof size
        fputs("  Proof: 2 G1 + 1 G2 = 128 bytes (compressed)\n", stderr)

        // Verify
        let verifier = Groth16Verifier()
        let verifyT0 = CFAbsoluteTimeGetCurrent()
        let valid = verifier.verify(proof: proof, vk: vk, publicInputs: pubInputs)
        let verifyTime = (CFAbsoluteTimeGetCurrent() - verifyT0) * 1000
        fputs("  Verify: \(String(format: "%.1f", verifyTime))ms -- \(valid ? "VALID" : "INVALID")\n", stderr)
    } catch {
        fputs("  Error: \(error)\n", stderr)
    }

    // 2. Benchmark at various sizes
    fputs("\n[2] Benchmark: multiplication chain circuits\n", stderr)
    let benchSizes = [8, 64, 256]

    for size in benchSizes {
        let (benchR1cs, benchPub, benchWit) = buildBenchCircuit(numConstraints: size)

        // Verify satisfaction
        var benchZ = [Fr](repeating: Fr.zero, count: benchR1cs.numVars)
        benchZ[0] = Fr.one
        benchZ[1] = benchPub[0]
        benchZ[2] = benchPub[1]
        for i in 0..<benchWit.count {
            benchZ[3 + i] = benchWit[i]
        }
        let benchSat = benchR1cs.isSatisfied(z: benchZ)
        if !benchSat {
            fputs("  n=\(size): R1CS NOT satisfied, skipping\n", stderr)
            continue
        }

        // Setup
        let sT0 = CFAbsoluteTimeGetCurrent()
        let (bPk, bVk) = setup.setup(r1cs: benchR1cs)
        let sTime = (CFAbsoluteTimeGetCurrent() - sT0) * 1000

        // Prove
        do {
            let prover = try Groth16Prover()

            let pT0 = CFAbsoluteTimeGetCurrent()
            let proof = try prover.prove(pk: bPk, r1cs: benchR1cs,
                                          publicInputs: benchPub, witness: benchWit)
            let pTime = (CFAbsoluteTimeGetCurrent() - pT0) * 1000

            // Verify
            let verifier = Groth16Verifier()
            let vT0 = CFAbsoluteTimeGetCurrent()
            let valid = verifier.verify(proof: proof, vk: bVk, publicInputs: benchPub)
            let vTime = (CFAbsoluteTimeGetCurrent() - vT0) * 1000

            fputs(String(format: "  n=%4d: setup %7.1fms | prove %7.1fms | verify %7.1fms | %@\n",
                        size, sTime, pTime, vTime, valid ? "VALID" : "INVALID"), stderr)
        } catch {
            fputs("  n=\(size): Error: \(error)\n", stderr)
        }
    }

    // 3. Pairing correctness test
    fputs("\n[3] BN254 Pairing sanity check\n", stderr)
    let pairingT0 = CFAbsoluteTimeGetCurrent()
    let g1 = bn254G1Generator()
    let g2 = bn254G2Generator()

    // e(g1, g2) should be non-trivial
    let pairing = bn254Pairing(g1, g2)
    let pairingTime = (CFAbsoluteTimeGetCurrent() - pairingT0) * 1000
    let isOne = fp12Equal(pairing, .one)
    fputs("  e(G1, G2) computed in \(String(format: "%.1f", pairingTime))ms\n", stderr)
    fputs("  e(G1, G2) == 1: \(isOne) (expected false for generator pairing)\n", stderr)

    // Bilinearity check: e(aG1, bG2) == e(G1, G2)^(ab)
    let a: UInt64 = 7
    let b: UInt64 = 11
    let aG1 = pointFromAffine(g1)
    let bG2 = g2FromAffine(g2)
    let aG1Scaled = pointScalarMul(aG1, frFromInt(a))
    let bG2Scaled = g2ScalarMul(bG2, [b, 0, 0, 0])

    if let aG1Aff = pointToAffine(aG1Scaled), let bG2Aff = g2ToAffine(bG2Scaled) {
        let lhs = bn254Pairing(aG1Aff, bG2Aff)  // e(aG1, bG2)

        // e(G1, G2)^(ab) via repeated squaring
        let ab = a * b  // 77
        var rhs = pairing
        var rhsPow = Fp12.one
        var k = ab
        while k > 0 {
            if k & 1 == 1 {
                rhsPow = fp12Mul(rhsPow, rhs)
            }
            rhs = fp12Sqr(rhs)
            k >>= 1
        }

        let bilinear = fp12Equal(lhs, rhsPow)
        fputs("  Bilinearity e(7G1, 11G2) == e(G1,G2)^77: \(bilinear)\n", stderr)
    }
}
