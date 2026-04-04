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
        // Debug: check NTT round-trip and H polynomial
        fputs("  [Debug] Checking NTT round-trip and H...\n", stderr)
        do {
            let prover2 = try Groth16Prover()
            // NTT round-trip test
            let testVec: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
            let fwd = try prover2.ntt.ntt(testVec)
            let back = try prover2.ntt.intt(fwd)
            var nttOk = true
            for i in 0..<4 { if !frSub(testVec[i], back[i]).isZero { nttOk = false } }
            fputs("  [Debug] NTT round-trip: \(nttOk)\n", stderr)

            // Check R1CS evaluations
            let nP2 = r1cs.numPublic
            var zz = [Fr](repeating: .zero, count: r1cs.numVars)
            zz[0] = .one; for ii in 0..<nP2 { zz[1+ii] = pub[ii] }
            for ii in 0..<wit.count { zz[1+nP2+ii] = wit[ii] }
            let az = r1cs.sparseMatVec(r1cs.aEntries, zz)
            let bz = r1cs.sparseMatVec(r1cs.bEntries, zz)
            let cz = r1cs.sparseMatVec(r1cs.cEntries, zz)
            var pointwiseOk = true
            for ii in 0..<r1cs.numConstraints {
                if !frSub(frMul(az[ii], bz[ii]), cz[ii]).isZero { pointwiseOk = false }
            }
            fputs("  [Debug] Az*Bz==Cz pointwise: \(pointwiseOk)\n", stderr)

            // Check H: compute H and verify A*B - C = H*Z at a random point
            let m = r1cs.numConstraints
            var domN2 = 1; var logD2 = 0
            while domN2 < m { domN2 <<= 1; logD2 += 1 }

            // Get A,B,C coefficients via INTT of evaluations
            var aEv = [Fr](repeating: .zero, count: domN2)
            var bEv = [Fr](repeating: .zero, count: domN2)
            var cEv = [Fr](repeating: .zero, count: domN2)
            for ii in 0..<m { aEv[ii] = az[ii]; bEv[ii] = bz[ii]; cEv[ii] = cz[ii] }
            let aCo = try prover2.ntt.intt(aEv)
            let bCo = try prover2.ntt.intt(bEv)
            let cCo = try prover2.ntt.intt(cEv)

            // Evaluate A, B, C at test point x=7
            let testX = frFromInt(7)
            func evalPoly(_ coeffs: [Fr], _ x: Fr) -> Fr {
                var result = Fr.zero; var xpow = Fr.one
                for c in coeffs { result = frAdd(result, frMul(c, xpow)); xpow = frMul(xpow, x) }
                return result
            }
            let aAt7 = evalPoly(aCo, testX)
            let bAt7 = evalPoly(bCo, testX)
            let cAt7 = evalPoly(cCo, testX)
            let pAt7 = frSub(frMul(aAt7, bAt7), cAt7)

            // Z(7) = 7^domN - 1
            var z7 = Fr.one; for _ in 0..<domN2 { z7 = frMul(z7, testX) }
            z7 = frSub(z7, .one)

            // Compute H using the same code as prover
            // Replicate computeH logic
            let bigN = domN2 * 2
            var aPad = [Fr](repeating: .zero, count: bigN)
            var bPad = [Fr](repeating: .zero, count: bigN)
            var cPad = [Fr](repeating: .zero, count: bigN)
            for ii in 0..<domN2 { aPad[ii] = aCo[ii]; bPad[ii] = bCo[ii]; cPad[ii] = cCo[ii] }
            let aEE = try prover2.ntt.ntt(aPad)
            let bEE = try prover2.ntt.ntt(bPad)
            let cEE = try prover2.ntt.ntt(cPad)
            var pEE = [Fr](repeating: .zero, count: bigN)
            for ii in 0..<bigN { pEE[ii] = frSub(frMul(aEE[ii], bEE[ii]), cEE[ii]) }
            let minusTwo = frNeg(frAdd(.one, .one))
            let minusTwoInv = frInverse(minusTwo)
            var hEE = [Fr](repeating: .zero, count: bigN)
            for ii in 0..<bigN {
                if ii % 2 == 0 { hEE[ii] = .zero }
                else { hEE[ii] = frMul(pEE[ii], minusTwoInv) }
            }
            let hCo = try prover2.ntt.intt(hEE)
            let hAt7 = evalPoly(Array(hCo.prefix(domN2)), testX)

            // Check: P(7) == H(7) * Z(7)
            let hz7 = frMul(hAt7, z7)
            fputs("  [Debug] P(7)==H(7)*Z(7): \(frSub(pAt7, hz7).isZero)\n", stderr)
            fputs("  [Debug] P(7) nonzero: \(!pAt7.isZero)\n", stderr)
            fputs("  [Debug] H(7) nonzero: \(!hAt7.isZero)\n", stderr)

            // Check even evaluations of P are zero
            var evenZeroCount = 0; var evenNonzeroCount = 0
            for ii in stride(from: 0, to: bigN, by: 2) {
                if pEE[ii].isZero { evenZeroCount += 1 } else { evenNonzeroCount += 1 }
            }
            fputs("  [Debug] P evals at even indices: \(evenZeroCount) zero, \(evenNonzeroCount) nonzero\n", stderr)

            // Check odd evaluations nonzero
            var oddNonzero = 0
            for ii in stride(from: 1, to: bigN, by: 2) {
                if !pEE[ii].isZero { oddNonzero += 1 }
            }
            fputs("  [Debug] P evals at odd indices: \(oddNonzero) nonzero out of \(bigN/2)\n", stderr)

            // Alternative: do polynomial long division of P by Z = x^domN - 1
            // P has degree < 2*domN, Z has degree domN
            // P = H * Z + R where deg(R) < domN
            // Since P vanishes on the N-th roots, R should be 0
            var pCoeffs = try prover2.ntt.intt(pEE)  // coefficients of P
            // Long division: P / (x^domN - 1)
            // H[i] = P[i + domN], and subtract: P[i] += H[i] (since Z = x^N - 1, so x^N * H[i] gives H[i] at position i+N, and -H[i] at position i)
            var hLong = [Fr](repeating: .zero, count: domN2)
            var rem = Array(pCoeffs.prefix(bigN))
            for ii in stride(from: bigN - 1, through: domN2, by: -1) {
                let q = rem[ii]  // quotient coefficient
                hLong[ii - domN2] = q
                rem[ii] = .zero
                rem[ii - domN2] = frAdd(rem[ii - domN2], q)  // subtract -1 * q = add q
            }
            // Check remainder is zero
            var remZero = true
            for ii in 0..<domN2 { if !rem[ii].isZero { remZero = false; break } }
            fputs("  [Debug] Long division remainder zero: \(remZero)\n", stderr)

            // Evaluate H from long division at x=7
            let hLongAt7 = evalPoly(hLong, testX)
            let hzLong7 = frMul(hLongAt7, z7)
            fputs("  [Debug] P(7)==H_longdiv(7)*Z(7): \(frSub(pAt7, hzLong7).isZero)\n", stderr)
        } catch {
            fputs("  [Debug] Error: \(error)\n", stderr)
        }
        // Debug: manually check the pairing equation
        fputs("  [Debug] Checking pairing equation manually...\n", stderr)
        let pA = pointToAffine(proof.a)!
        let pC = pointToAffine(proof.c)!
        let pB = g2ToAffine(proof.b)!
        let al = pointToAffine(vk.alpha_g1)!
        let be = g2ToAffine(vk.beta_g2)!
        let ga = g2ToAffine(vk.gamma_g2)!
        let de = g2ToAffine(vk.delta_g2)!
        var vkX = vk.ic[0]
        for i in 0..<pub.count {
            if !pub[i].isZero { vkX = pointAdd(vkX, pointScalarMul(vk.ic[i+1], pub[i])) }
        }
        let vx = pointToAffine(vkX)!
        // Check each pairing individually
        let negA = pointNegateAffine(pA)
        let ml1 = bn254MillerLoop(negA, pB)
        let ml2 = bn254MillerLoop(al, be)
        let ml3 = bn254MillerLoop(vx, ga)
        let ml4 = bn254MillerLoop(pC, de)
        let prod = fp12Mul(fp12Mul(ml1, ml2), fp12Mul(ml3, ml4))
        let result = bn254FinalExponentiation(prod)
        fputs("  [Debug] Product of Miller loops -> final exp == 1: \(fp12Equal(result, .one))\n", stderr)
        // Alternative: check e(A,B) == e(alpha,beta) * e(vkX,gamma) * e(C,delta)
        let eAB = bn254Pairing(pA, pB)
        let eAlBe = bn254Pairing(al, be)
        let eVxGa = bn254Pairing(vx, ga)
        let eCDe = bn254Pairing(pC, de)
        let rhs2 = fp12Mul(fp12Mul(eAlBe, eVxGa), eCDe)
        fputs("  [Debug] e(A,B) == e(al,be)*e(vx,ga)*e(C,de): \(fp12Equal(eAB, rhs2))\n", stderr)
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
    // Check G2 generator on curve: y^2 = x^3 + 3/xi, xi=9+u
    let g2gen = bn254G2Generator()
    let g2x = g2gen.x; let g2y = g2gen.y
    let g2y2 = fp2Sqr(g2y)
    let g2x3 = fp2Mul(fp2Sqr(g2x), g2x)
    let xi = Fp2(c0: fpFromInt(9), c1: fpFromInt(1))
    let xiInv = fp2Inverse(xi)
    let bTwist = fp2Mul(Fp2(c0: fpFromInt(3), c1: .zero), xiInv)
    let g2rhs = fp2Add(g2x3, bTwist)
    let g2OnCurve = fp2Sub(g2y2, g2rhs)
    fputs("  G2 on curve: c0=\(g2OnCurve.c0.isZero), c1=\(g2OnCurve.c1.isZero)\n", stderr)

    // Check G1 on curve
    let g1gen = bn254G1Generator()
    let g1y2 = fpSqr(g1gen.y)
    let g1x3p3 = fpAdd(fpMul(fpSqr(g1gen.x), g1gen.x), fpFromInt(3))
    fputs("  G1 on curve: \(fpSub(g1y2, g1x3p3).isZero)\n", stderr)

    // Check 2*G2 works
    let g2Proj = g2FromAffine(g2gen)
    let g2Dbl = g2Double(g2Proj)
    let g2DblA = g2ToAffine(g2Dbl)!
    let g2DblY2 = fp2Sqr(g2DblA.y)
    let g2DblRhs = fp2Add(fp2Mul(fp2Sqr(g2DblA.x), g2DblA.x), bTwist)
    let g2DblCheck = fp2Sub(g2DblY2, g2DblRhs)
    fputs("  2*G2 on curve: c0=\(g2DblCheck.c0.isZero), c1=\(g2DblCheck.c1.isZero)\n", stderr)

    let pT0 = CFAbsoluteTimeGetCurrent()
    let pair = bn254Pairing(bn254G1Generator(), bn254G2Generator())
    fputs("  e(G1,G2) in \(String(format: "%.1f", (CFAbsoluteTimeGetCurrent()-pT0)*1000))ms, ==1: \(fp12Equal(pair, .one))\n", stderr)

    // Check e(G1,G2)^r == 1 (where r is the group order)
    // If pairing works, e(G1,G2) should be non-trivial (not 1) but have order r

    // Bilinearity check: e(aP, Q) == e(P, aQ)
    let a = frFromInt(7)
    let aG1 = pointToAffine(pointScalarMul(pointFromAffine(bn254G1Generator()), a))!
    let aG2 = g2ToAffine(g2ScalarMul(g2FromAffine(bn254G2Generator()), frToInt(a)))!
    let lhs = bn254Pairing(aG1, bn254G2Generator())
    let rhs = bn254Pairing(bn254G1Generator(), aG2)
    fputs("  Bilinear e(7G,H)==e(G,7H): \(fp12Equal(lhs, rhs) ? "PASS" : "FAIL")\n", stderr)

    // Simple check: e(G, -H) * e(G, H) == 1 ?
    let negH = g2NegateAffine(bn254G2Generator())
    let eProd = fp12Mul(bn254MillerLoop(bn254G1Generator(), bn254G2Generator()),
                         bn254MillerLoop(bn254G1Generator(), negH))
    let eProdFinal = bn254FinalExponentiation(eProd)
    fputs("  e(G,H)*e(G,-H)==1: \(fp12Equal(eProdFinal, .one) ? "PASS" : "FAIL")\n", stderr)

    // Debug G2 scalar mul
    let g2P = g2FromAffine(bn254G2Generator())
    let g2_7a = g2ToAffine(g2ScalarMul(g2P, frToInt(frFromInt(7))))!
    var g2_7b = g2P
    for _ in 1..<7 { g2_7b = g2Add(g2_7b, g2P) }
    let g2_7bA = g2ToAffine(g2_7b)!
    let xMatch = fp2Sub(g2_7a.x, g2_7bA.x)
    let yMatch = fp2Sub(g2_7a.y, g2_7bA.y)
    fputs("  G2: 7*G==G+G+...+G: x=\(xMatch.c0.isZero && xMatch.c1.isZero), y=\(yMatch.c0.isZero && yMatch.c1.isZero)\n", stderr)

    // Check: e(2G, H) == e(G, H)^2
    let g1_2 = pointToAffine(pointDouble(pointFromAffine(bn254G1Generator())))!
    let e2g_h = bn254Pairing(g1_2, bn254G2Generator())
    let eg_h = bn254Pairing(bn254G1Generator(), bn254G2Generator())
    let eg_h_sq = fp12Mul(eg_h, eg_h)
    fputs("  e(2G,H)==e(G,H)^2: \(fp12Equal(e2g_h, eg_h_sq) ? "PASS" : "FAIL")\n", stderr)

    // Check: e(G, 2H) == e(G, H)^2
    let h2 = g2ToAffine(g2Double(g2FromAffine(bn254G2Generator())))!
    let eg_2h = bn254Pairing(bn254G1Generator(), h2)
    fputs("  e(G,2H)==e(G,H)^2: \(fp12Equal(eg_2h, eg_h_sq) ? "PASS" : "FAIL")\n", stderr)

    // Check Miller without correction: e_raw(2G,H) vs e_raw(G,H)^2
    let rawMiller = bn254MillerLoopNoCorrection(bn254G1Generator(), bn254G2Generator())
    let rawMillerFinal = bn254FinalExponentiation(rawMiller)
    let rawMiller2 = bn254MillerLoopNoCorrection(g1_2, bn254G2Generator())
    let rawMiller2Final = bn254FinalExponentiation(rawMiller2)
    let rawSq = fp12Mul(rawMillerFinal, rawMillerFinal)
    fputs("  e_noCorr(2G,H)==e_noCorr(G,H)^2: \(fp12Equal(rawMiller2Final, rawSq) ? "PASS" : "FAIL")\n", stderr)

    // Sanity: Fp2 mul check: (1+2u)*(3+4u) = (1*3-2*4)+(1*4+2*3)u = -5+10u
    let testA = Fp2(c0: fpFromInt(1), c1: fpFromInt(2))
    let testB = Fp2(c0: fpFromInt(3), c1: fpFromInt(4))
    let testC = fp2Mul(testA, testB)
    let expectedC0 = fpFromHex("0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd42")  // p-5
    fputs("  Fp2 mul c0==(-5): \(fpSub(testC.c0, expectedC0).isZero)\n", stderr)
    fputs("  Fp2 mul c1==(10): \(fpSub(testC.c1, fpFromInt(10)).isZero)\n", stderr)

    // Trace: check if Q1 is on the twist curve
    let g2gen2 = bn254G2Generator()
    let q1x = fp2Mul(fp2Conjugate(g2gen2.x), bn254_gamma_1_2_pub())
    let q1y = fp2Mul(fp2Conjugate(g2gen2.y), bn254_gamma_1_3_pub())
    let q1 = G2AffinePoint(x: q1x, y: q1y)
    let q1y2 = fp2Sqr(q1.y)
    let q1x3 = fp2Mul(fp2Sqr(q1.x), q1.x)
    let q1rhs = fp2Add(q1x3, bTwist)
    let q1check = fp2Sub(q1y2, q1rhs)
    fputs("  Q1=pi(G2) on twist: c0=\(q1check.c0.isZero), c1=\(q1check.c1.isZero)\n", stderr)
}
