import Foundation
import Metal
import zkMetal

public func runGroth16Bench() {
    fputs("\n--- Groth16 SNARK Benchmark (BN254) ---\n", stderr)
    fputs("\n[1] Example circuit: x^3 + x + 5 = y\n", stderr)
    let r1cs = buildExampleCircuit()
    let (pub, wit) = computeExampleWitness(x: 3)
    var z = [Fr](repeating: .zero, count: r1cs.numVars)
    z[0] = .one; z[1] = pub[0]; z[2] = pub[1]
    for i in 0..<wit.count { z[3+i] = wit[i] }
    fputs("  R1CS satisfied: \(r1cs.isSatisfied(z: z))\n", stderr)
    let setup = Groth16Setup()
    let t0 = CFAbsoluteTimeGetCurrent()
    let (pk, vk) = setup.setup(r1cs: r1cs)
    fputs("  Setup: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent()-t0)*1000))ms\n", stderr)
    do {
        let prover = try Groth16Prover()
        let pt = CFAbsoluteTimeGetCurrent()
        let proof = try prover.prove(pk: pk, r1cs: r1cs, publicInputs: pub, witness: wit)
        fputs("  Prove: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent()-pt)*1000))ms\n", stderr)
        let verifier = Groth16Verifier()
        let vt = CFAbsoluteTimeGetCurrent()
        let valid = verifier.verify(proof: proof, vk: vk, publicInputs: pub)
        fputs("  Verify: \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent()-vt)*1000))ms -- \(valid ? "VALID" : "INVALID")\n", stderr)
    } catch { fputs("  Error: \(error)\n", stderr) }
    fputs("\n[2] Bench circuits\n", stderr)
    for sz in [8, 64, 256] {
        let (br, bp, bw) = buildBenchCircuit(numConstraints: sz)
        var bz = [Fr](repeating: .zero, count: br.numVars)
        bz[0] = .one; bz[1] = bp[0]; bz[2] = bp[1]
        for i in 0..<bw.count { bz[3+i] = bw[i] }
        guard br.isSatisfied(z: bz) else { fputs("  n=\(sz): FAIL\n", stderr); continue }
        let st = CFAbsoluteTimeGetCurrent(); let (bPk, bVk) = setup.setup(r1cs: br)
        let sT = (CFAbsoluteTimeGetCurrent()-st)*1000
        do {
            let prover = try Groth16Prover()
            prover.profileGroth16 = (sz == 256)
            let pt = CFAbsoluteTimeGetCurrent()
            let proof = try prover.prove(pk: bPk, r1cs: br, publicInputs: bp, witness: bw)
            let pT = (CFAbsoluteTimeGetCurrent()-pt)*1000
            let verifier = Groth16Verifier()
            let vt = CFAbsoluteTimeGetCurrent()
            let valid = verifier.verify(proof: proof, vk: bVk, publicInputs: bp)
            let vT = (CFAbsoluteTimeGetCurrent()-vt)*1000
            fputs(String(format: "  n=%4d: setup %7.1fms | prove %7.1fms | verify %7.1fms | %@\n",
                        sz, sT, pT, vT, valid ? "VALID" : "INVALID"), stderr)
        } catch { fputs("  n=\(sz): \(error)\n", stderr) }
    }
    fputs("\n[3] BN254 Pairing Checks\n", stderr)
    let g1gen = bn254G1Generator()
    let g2gen = bn254G2Generator()
    let pT0 = CFAbsoluteTimeGetCurrent()
    let pair = bn254Pairing(g1gen, g2gen)
    fputs("  e(G1,G2) in \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent()-pT0)*1000))ms, ==1: \(fp12Equal(pair, .one))\n", stderr)

    let a = frFromInt(7)
    let aG1 = pointToAffine(pointScalarMul(pointFromAffine(g1gen), a))!
    let aG2 = g2ToAffine(g2ScalarMul(g2FromAffine(g2gen), frToInt(a)))!
    let lhs = bn254Pairing(aG1, g2gen)
    let rhs = bn254Pairing(g1gen, aG2)
    fputs("  Bilinear e(7G,H)==e(G,7H): \(fp12Equal(lhs, rhs) ? "PASS" : "FAIL")\n", stderr)

    let negH = g2NegateAffine(g2gen)
    let eProd = fp12Mul(bn254MillerLoop(g1gen, g2gen), bn254MillerLoop(g1gen, negH))
    let eProdFinal = bn254FinalExponentiation(eProd)
    fputs("  e(G,H)*e(G,-H)==1: \(fp12Equal(eProdFinal, .one) ? "PASS" : "FAIL")\n", stderr)

    let g1_2 = pointToAffine(pointDouble(pointFromAffine(g1gen)))!
    let e2g_h = bn254Pairing(g1_2, g2gen)
    let eg_h_sq = fp12Mul(pair, pair)
    fputs("  e(2G,H)==e(G,H)^2: \(fp12Equal(e2g_h, eg_h_sq) ? "PASS" : "FAIL")\n", stderr)

    let h2 = g2ToAffine(g2Double(g2FromAffine(g2gen)))!
    let eg_2h = bn254Pairing(g1gen, h2)
    fputs("  e(G,2H)==e(G,H)^2: \(fp12Equal(eg_2h, eg_h_sq) ? "PASS" : "FAIL")\n", stderr)
}
