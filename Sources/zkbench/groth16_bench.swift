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

        // Debug: scalar-level Groth16 equation check
        if !valid {
            fputs("  [Debug] Scalar-level Groth16 check...\n", stderr)
            scalarGroth16Check(r1cs: r1cs, publicInputs: pub, witness: wit)
        }
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
    fputs("\n[3] BN254 Pairing Debug\n", stderr)
    let g1gen = bn254G1Generator()
    let g2gen = bn254G2Generator()
    let pT0 = CFAbsoluteTimeGetCurrent()
    let pair = bn254Pairing(g1gen, g2gen)
    fputs("  e(G1,G2) in \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent()-pT0)*1000))ms, ==1: \(fp12Equal(pair, .one))\n", stderr)

    // Bilinearity check: e(aP, Q) == e(P, aQ)
    let a = frFromInt(7)
    let aG1 = pointToAffine(pointScalarMul(pointFromAffine(g1gen), a))!
    let aG2 = g2ToAffine(g2ScalarMul(g2FromAffine(g2gen), frToInt(a)))!
    let lhs = bn254Pairing(aG1, g2gen)
    let rhs = bn254Pairing(g1gen, aG2)
    fputs("  Bilinear e(7G,H)==e(G,7H): \(fp12Equal(lhs, rhs) ? "PASS" : "FAIL")\n", stderr)

    // e(G, -H) * e(G, H) == 1
    let negH = g2NegateAffine(g2gen)
    let eProd = fp12Mul(bn254MillerLoop(g1gen, g2gen), bn254MillerLoop(g1gen, negH))
    let eProdFinal = bn254FinalExponentiation(eProd)
    fputs("  e(G,H)*e(G,-H)==1: \(fp12Equal(eProdFinal, .one) ? "PASS" : "FAIL")\n", stderr)

    // e(2G, H) == e(G, H)^2
    let g1_2 = pointToAffine(pointDouble(pointFromAffine(g1gen)))!
    let e2g_h = bn254Pairing(g1_2, g2gen)
    let eg_h_sq = fp12Mul(pair, pair)
    fputs("  e(2G,H)==e(G,H)^2: \(fp12Equal(e2g_h, eg_h_sq) ? "PASS" : "FAIL")\n", stderr)

    // e(G, 2H) == e(G, H)^2
    let h2 = g2ToAffine(g2Double(g2FromAffine(g2gen)))!
    let eg_2h = bn254Pairing(g1gen, h2)
    fputs("  e(G,2H)==e(G,H)^2: \(fp12Equal(eg_2h, eg_h_sq) ? "PASS" : "FAIL")\n", stderr)
}

/// Scalar-level check of the Groth16 equation (no pairings, just Fr arithmetic)
func scalarGroth16Check(r1cs: R1CSInstance, publicInputs: [Fr], witness: [Fr]) {
    let nPub = r1cs.numPublic; let m = r1cs.numConstraints; let numV = r1cs.numVars
    var domN = 1; var logD = 0
    while domN < m { domN <<= 1; logD += 1 }

    // Build z vector
    var zz = [Fr](repeating: .zero, count: numV)
    zz[0] = .one; for i in 0..<nPub { zz[1+i] = publicInputs[i] }
    for i in 0..<witness.count { zz[1+nPub+i] = witness[i] }

    // Use deterministic toxic waste for reproducibility
    let tau = frFromInt(13); let alpha = frFromInt(5); let beta = frFromInt(7)
    let gamma = frFromInt(11); let delta = frFromInt(3)
    let gammaInv = frInverse(gamma); let deltaInv = frInverse(delta)

    // Compute omega and Lagrange basis at tau
    let omega = frRootOfUnity(logN: logD)
    var omegaPow = [Fr](repeating: .one, count: domN)
    for i in 1..<domN { omegaPow[i] = frMul(omegaPow[i-1], omega) }

    var zTau = Fr.one; for _ in 0..<domN { zTau = frMul(zTau, tau) }
    zTau = frSub(zTau, .one)
    let nFr = frFromInt(UInt64(domN))
    let zOverN = frMul(zTau, frInverse(nFr))
    var lagAtTau = [Fr](repeating: .zero, count: domN)
    for i in 0..<domN {
        lagAtTau[i] = frMul(frMul(zOverN, omegaPow[i]), frInverse(frSub(tau, omegaPow[i])))
    }

    // u_j(tau), v_j(tau), w_j(tau)
    var uT = [Fr](repeating: .zero, count: numV)
    var vT = [Fr](repeating: .zero, count: numV)
    var wT = [Fr](repeating: .zero, count: numV)
    for e in r1cs.aEntries { uT[e.col] = frAdd(uT[e.col], frMul(e.val, lagAtTau[e.row])) }
    for e in r1cs.bEntries { vT[e.col] = frAdd(vT[e.col], frMul(e.val, lagAtTau[e.row])) }
    for e in r1cs.cEntries { wT[e.col] = frAdd(wT[e.col], frMul(e.val, lagAtTau[e.row])) }

    // Scalar sums
    var sumU = Fr.zero; var sumV = Fr.zero; var sumW = Fr.zero
    for j in 0..<numV {
        sumU = frAdd(sumU, frMul(zz[j], uT[j]))
        sumV = frAdd(sumV, frMul(zz[j], vT[j]))
        sumW = frAdd(sumW, frMul(zz[j], wT[j]))
    }

    // Compute H(tau) via polynomial manipulation
    do {
        let prover2 = try Groth16Prover()
        let az = r1cs.sparseMatVec(r1cs.aEntries, zz)
        let bz = r1cs.sparseMatVec(r1cs.bEntries, zz)
        let cz = r1cs.sparseMatVec(r1cs.cEntries, zz)
        var aEv = [Fr](repeating: .zero, count: domN)
        var bEv = [Fr](repeating: .zero, count: domN)
        var cEv = [Fr](repeating: .zero, count: domN)
        for i in 0..<m { aEv[i] = az[i]; bEv[i] = bz[i]; cEv[i] = cz[i] }
        let aCo = try prover2.ntt.intt(aEv)
        let bCo = try prover2.ntt.intt(bEv)
        let cCo = try prover2.ntt.intt(cEv)
        let bigN = domN * 2
        var aPad = [Fr](repeating: .zero, count: bigN)
        var bPad = [Fr](repeating: .zero, count: bigN)
        var cPad = [Fr](repeating: .zero, count: bigN)
        for i in 0..<domN { aPad[i] = aCo[i]; bPad[i] = bCo[i]; cPad[i] = cCo[i] }
        let aEE = try prover2.ntt.ntt(aPad)
        let bEE = try prover2.ntt.ntt(bPad)
        let cEE = try prover2.ntt.ntt(cPad)
        var pEE = [Fr](repeating: .zero, count: bigN)
        for i in 0..<bigN { pEE[i] = frSub(frMul(aEE[i], bEE[i]), cEE[i]) }
        let pCoeffs = try prover2.ntt.intt(pEE)
        var hCoeffs = [Fr](repeating: .zero, count: domN)
        var rem2 = Array(pCoeffs.prefix(bigN))
        for i in stride(from: bigN - 1, through: domN, by: -1) {
            let q = rem2[i]; hCoeffs[i - domN] = q; rem2[i] = .zero
            rem2[i - domN] = frAdd(rem2[i - domN], q)
        }
        // Evaluate H at tau
        func evalPoly(_ coeffs: [Fr], _ x: Fr) -> Fr {
            var result = Fr.zero; var xpow = Fr.one
            for c in coeffs { result = frAdd(result, frMul(c, xpow)); xpow = frMul(xpow, x) }
            return result
        }
        let hAtTau = evalPoly(hCoeffs, tau)

        // Check Lagrange basis: sum L_i(tau) should be 1
        var lagSum = Fr.zero
        for i in 0..<domN { lagSum = frAdd(lagSum, lagAtTau[i]) }
        fputs("    sum L_i(tau) == 1: \(frSub(lagSum, .one).isZero)\n", stderr)

        // Check A(tau) via Lagrange vs coefficient evaluation
        let aAtTauCoeff = evalPoly(aCo, tau)
        fputs("    A(tau) Lagrange==Coeff: \(frSub(sumU, aAtTauCoeff).isZero)\n", stderr)
        fputs("    sumU (Lagrange): nonzero=\(!sumU.isZero)\n", stderr)
        fputs("    A(tau) (Coeff): nonzero=\(!aAtTauCoeff.isZero)\n", stderr)

        // Check: sumU * sumV - sumW == H(tau) * Z(tau)
        let uvMinusW = frSub(frMul(sumU, sumV), sumW)
        let hzCheck = frMul(hAtTau, zTau)
        fputs("    sumU*sumV - sumW == H*Z: \(frSub(uvMinusW, hzCheck).isZero)\n", stderr)

        // Also check with coefficient-evaluated A, B, C
        let bAtTauCoeff = evalPoly(bCo, tau)
        let cAtTauCoeff = evalPoly(cCo, tau)
        let abMinusCCoeff = frSub(frMul(aAtTauCoeff, bAtTauCoeff), cAtTauCoeff)
        fputs("    A(t)*B(t)-C(t) [coeff] == H*Z: \(frSub(abMinusCCoeff, hzCheck).isZero)\n", stderr)

        // Check full Groth16 equation (r=0, s=0):
        // A = alpha + sumU, B = beta + sumV
        // C = sum_witness(z_j * (beta*u_j + alpha*v_j + w_j)/delta) + H*Z/delta
        // vkX = sum_public(z_j * (beta*u_j + alpha*v_j + w_j)/gamma)
        // A*B == alpha*beta + vkX*gamma + C*delta
        let aScalar = frAdd(alpha, sumU)
        let bScalar = frAdd(beta, sumV)

        var cScalar = Fr.zero
        for j in (nPub+1)..<numV {
            let coeff = frMul(frAdd(frAdd(frMul(beta, uT[j]), frMul(alpha, vT[j])), wT[j]), deltaInv)
            cScalar = frAdd(cScalar, frMul(zz[j], coeff))
        }
        cScalar = frAdd(cScalar, frMul(hAtTau, frMul(zTau, deltaInv)))

        var vkxScalar = Fr.zero
        for j in 0...(nPub) {
            let coeff = frMul(frAdd(frAdd(frMul(beta, uT[j]), frMul(alpha, vT[j])), wT[j]), gammaInv)
            vkxScalar = frAdd(vkxScalar, frMul(zz[j], coeff))
        }

        let lhsS = frMul(aScalar, bScalar)
        let rhsS = frAdd(frAdd(frMul(alpha, beta), frMul(vkxScalar, gamma)), frMul(cScalar, delta))
        fputs("    Scalar A*B == a*b + vkX*g + C*d: \(frSub(lhsS, rhsS).isZero)\n", stderr)
    } catch {
        fputs("    Error: \(error)\n", stderr)
    }
}
