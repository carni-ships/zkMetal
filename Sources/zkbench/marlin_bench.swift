// marlin_bench — Benchmark and correctness test for Marlin preprocessed SNARK
//
// Tests:
// [1] Correctness: setup + prove + verify for example R1CS (x^3 + x + 5 = y)
// [2] Performance at various constraint counts (16, 64, 256, 1024)
// [3] Comparison with Groth16 at matching sizes

import Foundation
import zkMetal

public func runMarlinBench() {
    fputs("\n--- Marlin Preprocessed SNARK Benchmark (BN254) ---\n", stderr)

    let srsSecret: [UInt32] = [0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222,
                                0x33333333, 0x44444444, 0x55555555, 0x00000001]
    let srsSecretFr = frFromLimbs(srsSecret)
    let generator = PointAffine(x: fpFromInt(1), y: fpFromInt(2))

    // ========== [1] Correctness ==========
    fputs("\n[1] Correctness: x^3 + x + 5 = y\n", stderr)

    let r1cs = buildExampleCircuit()
    let (pub, wit) = computeExampleWitness(x: 3)
    var z = [Fr](repeating: .zero, count: r1cs.numVars)
    z[0] = .one; z[1] = pub[0]; z[2] = pub[1]
    for i in 0..<wit.count { z[3 + i] = wit[i] }
    fputs("  R1CS satisfied: \(r1cs.isSatisfied(z: z))\n", stderr)

    do {
        let srsSize = 256
        let srs = KZGEngine.generateTestSRS(secret: srsSecret, size: srsSize, generator: generator)
        let kzg = try KZGEngine(srs: srs)
        let nttE = try NTTEngine()
        let engine = MarlinEngine(kzg: kzg, ntt: nttE)

        let t0 = CFAbsoluteTimeGetCurrent()
        let (pk, vk) = try engine.setup(r1cs: r1cs, srsSecret: srsSecretFr)
        let setupMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        fputs(String(format: "  Setup: %.1fms\n", setupMs), stderr)
        fputs("  Index: constraints=\(pk.index.numConstraints), vars=\(pk.index.numVariables), nnz=\(pk.index.numNonZero)\n", stderr)
        fputs("  Domains: H=\(pk.index.constraintDomainSize), K=\(pk.index.variableDomainSize), K_NZ=\(pk.index.nonZeroDomainSize)\n", stderr)

        let t1 = CFAbsoluteTimeGetCurrent()
        let proof = try engine.prove(r1cs: r1cs, publicInputs: pub, witness: wit, pk: pk)
        let proveMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000
        fputs(String(format: "  Prove: %.1fms\n", proveMs), stderr)

        let t2 = CFAbsoluteTimeGetCurrent()
        let valid = engine.verify(vk: vk, publicInput: pub, proof: proof)
        let verifyMs = (CFAbsoluteTimeGetCurrent() - t2) * 1000
        fputs(String(format: "  Verify: %.1fms -- %@\n", verifyMs, valid ? "VALID" : "INVALID"), stderr)

        // Reference: MarlinTestProver
        fputs("\n  MarlinTestProver (reference):\n", stderr)
        let testProver = MarlinTestProver(kzg: kzg)
        let t3 = CFAbsoluteTimeGetCurrent()
        let (tvk, tproof) = try testProver.generateTestProof(
            numConstraints: r1cs.numConstraints, publicInput: pub, srsSecret: srsSecretFr)
        let testProveMs = (CFAbsoluteTimeGetCurrent() - t3) * 1000
        let verifier = MarlinVerifier(kzg: kzg)
        let t4 = CFAbsoluteTimeGetCurrent()
        let tvalid = verifier.verify(vk: tvk, publicInput: pub, proof: tproof)
        let testVerifyMs = (CFAbsoluteTimeGetCurrent() - t4) * 1000
        fputs(String(format: "  TestProver: prove %.1fms, verify %.1fms -- %@\n",
                     testProveMs, testVerifyMs, tvalid ? "VALID" : "INVALID"), stderr)

    } catch {
        fputs("  Error: \(error)\n", stderr)
    }

    // ========== [2] Performance scaling ==========
    fputs("\n[2] Bench circuits (Marlin)\n", stderr)
    fputs("       n     setup     prove    verify  result\n", stderr)

    for sz in [16, 64, 256, 1024] {
        let (br, bp, bw) = buildBenchCircuit(numConstraints: sz)
        var bz = [Fr](repeating: .zero, count: br.numVars)
        bz[0] = .one; bz[1] = bp[0]; bz[2] = bp[1]
        for i in 0..<bw.count { bz[3 + i] = bw[i] }
        guard br.isSatisfied(z: bz) else {
            fputs(String(format: "  %6d  FAIL (R1CS not satisfied)\n", sz), stderr)
            continue
        }

        do {
            let maxDomain = marlinNextPow2(br.numVars) * 2
            let srsSize = max(maxDomain + 8, 128)
            let srs = KZGEngine.generateTestSRS(secret: srsSecret, size: srsSize, generator: generator)
            let kzg = try KZGEngine(srs: srs)
            let nttE = try NTTEngine()
            let engine = MarlinEngine(kzg: kzg, ntt: nttE)

            let st = CFAbsoluteTimeGetCurrent()
            let (pk, vk) = try engine.setup(r1cs: br, srsSecret: srsSecretFr)
            let setupT = (CFAbsoluteTimeGetCurrent() - st) * 1000

            let pt = CFAbsoluteTimeGetCurrent()
            let proof = try engine.prove(r1cs: br, publicInputs: bp, witness: bw, pk: pk)
            let proveT = (CFAbsoluteTimeGetCurrent() - pt) * 1000

            let vt = CFAbsoluteTimeGetCurrent()
            let valid = engine.verify(vk: vk, publicInput: bp, proof: proof)
            let verifyT = (CFAbsoluteTimeGetCurrent() - vt) * 1000

            fputs(String(format: "  %6d  %7.1fms  %7.1fms  %7.1fms  %@\n",
                         sz, setupT, proveT, verifyT, valid ? "VALID" : "INVALID"), stderr)
        } catch {
            fputs("  n=\(sz): Error: \(error)\n", stderr)
        }
    }

    // ========== [3] Comparison: Marlin vs Groth16 ==========
    fputs("\n[3] Marlin vs Groth16 comparison\n", stderr)
    fputs("       n |   M-setup   M-prove  M-verify |   G-setup   G-prove  G-verify\n", stderr)

    for sz in [16, 64, 256] {
        let (br, bp, bw) = buildBenchCircuit(numConstraints: sz)
        var bz = [Fr](repeating: .zero, count: br.numVars)
        bz[0] = .one; bz[1] = bp[0]; bz[2] = bp[1]
        for i in 0..<bw.count { bz[3 + i] = bw[i] }
        guard br.isSatisfied(z: bz) else { continue }

        do {
            let maxDomain = marlinNextPow2(br.numVars) * 2
            let srsSize = max(maxDomain + 8, 128)
            let srs = KZGEngine.generateTestSRS(secret: srsSecret, size: srsSize, generator: generator)
            let kzg = try KZGEngine(srs: srs)
            let nttE = try NTTEngine()
            let marlin = MarlinEngine(kzg: kzg, ntt: nttE)

            // Marlin
            let ms0 = CFAbsoluteTimeGetCurrent()
            let (mpk, mvk) = try marlin.setup(r1cs: br, srsSecret: srsSecretFr)
            let mSetup = (CFAbsoluteTimeGetCurrent() - ms0) * 1000

            let mp0 = CFAbsoluteTimeGetCurrent()
            let mproof = try marlin.prove(r1cs: br, publicInputs: bp, witness: bw, pk: mpk)
            let mProve = (CFAbsoluteTimeGetCurrent() - mp0) * 1000

            let mv0 = CFAbsoluteTimeGetCurrent()
            _ = marlin.verify(vk: mvk, publicInput: bp, proof: mproof)
            let mVerify = (CFAbsoluteTimeGetCurrent() - mv0) * 1000

            // Groth16
            let g16setup = Groth16Setup()
            let gs0 = CFAbsoluteTimeGetCurrent()
            let (gpk, gvk) = g16setup.setup(r1cs: br)
            let gSetup = (CFAbsoluteTimeGetCurrent() - gs0) * 1000

            let gprover = try Groth16Prover()
            let gp0 = CFAbsoluteTimeGetCurrent()
            let gproof = try gprover.prove(pk: gpk, r1cs: br, publicInputs: bp, witness: bw)
            let gProve = (CFAbsoluteTimeGetCurrent() - gp0) * 1000

            let gverifier = Groth16Verifier()
            let gv0 = CFAbsoluteTimeGetCurrent()
            _ = gverifier.verify(proof: gproof, vk: gvk, publicInputs: bp)
            let gVerify = (CFAbsoluteTimeGetCurrent() - gv0) * 1000

            fputs(String(format: "  %6d | %7.1fms  %7.1fms  %7.1fms | %7.1fms  %7.1fms  %7.1fms\n",
                         sz, mSetup, mProve, mVerify, gSetup, gProve, gVerify), stderr)
        } catch {
            fputs("  n=\(sz): Error: \(error)\n", stderr)
        }
    }

    fputs("\n--- Marlin Benchmark Complete ---\n", stderr)
}

private func marlinNextPow2(_ n: Int) -> Int {
    var p = 1
    while p < n { p *= 2 }
    return max(p, 2)
}
