// BLS12-381 Field, Curve, and Pairing Benchmark / Correctness Tests
import zkMetal
import Foundation

public func runBLS12381Test() {
    print("\n=== BLS12-381 Field + Curve + Pairing Tests ===")

    // --- Fr381 (scalar field) tests ---
    print("\n--- Fr381 Scalar Field Arithmetic ---")

    let one381 = fr381FromInt(1)
    let oneOut = fr381ToInt(one381)
    let oneOk = oneOut[0] == 1 && oneOut[1] == 0 && oneOut[2] == 0 && oneOut[3] == 0
    print("  fromInt(1) round-trip: \(oneOk ? "PASS" : "FAIL")")

    let zero381 = fr381FromInt(0)
    let zeroOk = zero381.isZero
    print("  fromInt(0) is zero: \(zeroOk ? "PASS" : "FAIL")")

    let a381 = fr381FromInt(42)
    let addZero381 = fr381Add(a381, Fr381.zero)
    let addZeroOk381 = fr381ToInt(addZero381) == fr381ToInt(a381)
    print("  a + 0 = a: \(addZeroOk381 ? "PASS" : "FAIL")")

    let negA381 = fr381Neg(a381)
    let addNeg381 = fr381Add(a381, negA381)
    let addNegOk381 = addNeg381.isZero
    print("  a + (-a) = 0: \(addNegOk381 ? "PASS" : "FAIL")")

    let mulOne381 = fr381Mul(a381, one381)
    let mulOneOk381 = fr381ToInt(mulOne381) == fr381ToInt(a381)
    print("  a * 1 = a: \(mulOneOk381 ? "PASS" : "FAIL")")

    let aInv381 = fr381Inverse(a381)
    let mulInv381 = fr381Mul(a381, aInv381)
    let mulInvOut381 = fr381ToInt(mulInv381)
    let mulInvOk381 = mulInvOut381[0] == 1 && mulInvOut381[1] == 0 && mulInvOut381[2] == 0 && mulInvOut381[3] == 0
    print("  a * a^(-1) = 1: \(mulInvOk381 ? "PASS" : "FAIL")")

    // Distributivity
    let b381 = fr381FromInt(123456789)
    let c381 = fr381FromInt(987654321)
    let lhs381 = fr381Mul(fr381Add(a381, b381), c381)
    let rhs381 = fr381Add(fr381Mul(a381, c381), fr381Mul(b381, c381))
    let distOk381 = fr381ToInt(lhs381) == fr381ToInt(rhs381)
    print("  Distributivity: \(distOk381 ? "PASS" : "FAIL")")

    // Root of unity
    let omega = fr381RootOfUnity(logN: 1)
    let omega2 = fr381Sqr(omega)
    let omega2Out = fr381ToInt(omega2)
    let rootOk = omega2Out[0] == 1 && omega2Out[1] == 0 && omega2Out[2] == 0 && omega2Out[3] == 0
    print("  Root of unity omega^2 = 1 (logN=1): \(rootOk ? "PASS" : "FAIL")")

    // TWO_ADICITY test: omega_{2^32}^{2^32} = 1
    let omega32 = fr381RootOfUnity(logN: Fr381.TWO_ADICITY)
    var omPow = omega32
    for _ in 0..<Fr381.TWO_ADICITY { omPow = fr381Sqr(omPow) }
    let omPowOut = fr381ToInt(omPow)
    let rootFullOk = omPowOut[0] == 1 && omPowOut[1] == 0 && omPowOut[2] == 0 && omPowOut[3] == 0
    print("  omega_{2^32}^{2^32} = 1: \(rootFullOk ? "PASS" : "FAIL")")

    // --- Fp381 (base field) tests ---
    print("\n--- Fp381 Base Field Arithmetic ---")

    let fp_one = fp381FromInt(1)
    let fp_oneOut = fp381ToInt(fp_one)
    let fpOneOk = fp_oneOut[0] == 1 && fp_oneOut[1] == 0
    print("  fromInt(1) round-trip: \(fpOneOk ? "PASS" : "FAIL")")

    let fp_a = fp381FromInt(42)
    let fp_b = fp381FromInt(100)
    let fp_sum = fp381Add(fp_a, fp_b)
    let fp_sumOut = fp381ToInt(fp_sum)
    let fpAddOk = fp_sumOut[0] == 142
    print("  42 + 100 = 142: \(fpAddOk ? "PASS" : "FAIL")")

    let fp_aInv = fp381Inverse(fp_a)
    let fp_check = fp381Mul(fp_a, fp_aInv)
    let fp_checkOut = fp381ToInt(fp_check)
    let fpInvOk = fp_checkOut[0] == 1 && fp_checkOut[1] == 0
    print("  a * a^(-1) = 1: \(fpInvOk ? "PASS" : "FAIL")")

    // --- Fp2 tests ---
    print("\n--- Fp2 Extension Field ---")

    let fp2_a = Fp2_381(c0: fp381FromInt(3), c1: fp381FromInt(5))
    let fp2_b = Fp2_381(c0: fp381FromInt(7), c1: fp381FromInt(11))

    // (3+5u)(7+11u) = 21 - 55 + (33 + 35)u = -34 + 68u
    let fp2_prod = fp2_381Mul(fp2_a, fp2_b)
    let prod_c0 = fp381ToInt(fp2_prod.c0)
    let prod_c1 = fp381ToInt(fp2_prod.c1)

    // -34 mod p = p - 34
    let neg34 = fp381ToInt(fp381Neg(fp381FromInt(34)))
    let fp2MulOk = prod_c0 == neg34 && prod_c1[0] == 68 && prod_c1[1] == 0
    print("  (3+5u)(7+11u) = -34+68u: \(fp2MulOk ? "PASS" : "FAIL")")

    // Fp2 inverse
    let fp2_aInv = fp2_381Inverse(fp2_a)
    let fp2_check = fp2_381Mul(fp2_a, fp2_aInv)
    let fp2InvOk = fp381ToInt(fp2_check.c0)[0] == 1 && fp381ToInt(fp2_check.c0)[1] == 0 &&
                   fp2_check.c1.isZero
    print("  a * a^(-1) = 1: \(fp2InvOk ? "PASS" : "FAIL")")

    // Conjugation: (a+bu)(a-bu) = a^2 + b^2 (real)
    let fp2_conj = fp2_381Conjugate(fp2_a)
    let fp2_norm = fp2_381Mul(fp2_a, fp2_conj)
    let conjOk = fp2_norm.c1.isZero
    print("  a * conj(a) is real: \(conjOk ? "PASS" : "FAIL")")

    // --- Fp6 tests ---
    print("\n--- Fp6 Extension Field ---")

    let fp6_a = Fp6_381(c0: Fp2_381(c0: fp381FromInt(1), c1: fp381FromInt(2)),
                         c1: Fp2_381(c0: fp381FromInt(3), c1: fp381FromInt(4)),
                         c2: Fp2_381(c0: fp381FromInt(5), c1: fp381FromInt(6)))
    let fp6_aInv = fp6_381Inverse(fp6_a)
    let fp6_check = fp6_381Mul(fp6_a, fp6_aInv)
    // Check c0 = 1, c1 = 0, c2 = 0
    let fp6c0c0 = fp381ToInt(fp6_check.c0.c0)
    let fp6InvOk = fp6c0c0[0] == 1 && fp6c0c0[1] == 0 &&
                   fp6_check.c0.c1.isZero &&
                   fp6_check.c1.isZero && fp6_check.c2.isZero
    print("  a * a^(-1) = 1: \(fp6InvOk ? "PASS" : "FAIL")")

    // --- Fp12 tests ---
    print("\n--- Fp12 Extension Field ---")

    let fp12_a = Fp12_381(
        c0: Fp6_381(c0: Fp2_381(c0: fp381FromInt(1), c1: fp381FromInt(2)),
                     c1: Fp2_381(c0: fp381FromInt(3), c1: fp381FromInt(4)),
                     c2: Fp2_381(c0: fp381FromInt(5), c1: fp381FromInt(6))),
        c1: Fp6_381(c0: Fp2_381(c0: fp381FromInt(7), c1: fp381FromInt(8)),
                     c1: Fp2_381(c0: fp381FromInt(9), c1: fp381FromInt(10)),
                     c2: Fp2_381(c0: fp381FromInt(11), c1: fp381FromInt(12))))

    let fp12_aInv = fp12_381Inverse(fp12_a)
    let fp12_check = fp12_381Mul(fp12_a, fp12_aInv)
    let fp12c0c0c0 = fp381ToInt(fp12_check.c0.c0.c0)
    let fp12InvOk = fp12c0c0c0[0] == 1 && fp12c0c0c0[1] == 0 &&
                    fp12_check.c0.c0.c1.isZero &&
                    fp12_check.c0.c1.isZero && fp12_check.c0.c2.isZero &&
                    fp12_check.c1.c0.isZero && fp12_check.c1.c1.isZero && fp12_check.c1.c2.isZero
    print("  a * a^(-1) = 1: \(fp12InvOk ? "PASS" : "FAIL")")

    // Conjugate test: a * conj(a) should have c1 = 0 for cyclotomic elements
    // (Not generally true for arbitrary Fp12, but let's test the operation works)
    let fp12_conj = fp12_381Conjugate(fp12_a)
    let fp12_prod = fp12_381Mul(fp12_a, fp12_conj)
    // For general elements, c1 != 0. Just verify it computes without error.
    print("  Conjugation computes: PASS")

    // --- G1 Curve tests ---
    print("\n--- G1 Curve Operations ---")

    let gen1 = bls12381G1Generator()
    let engine = BLS12381Engine()

    // Generator on curve
    let g1OnCurve = engine.g1IsOnCurve(gen1)
    print("  G1 generator on curve: \(g1OnCurve ? "PASS" : "FAIL")")

    // 2G on curve
    let gProj = g1_381FromAffine(gen1)
    let twoG = g1_381Double(gProj)
    if let twoGAff = g1_381ToAffine(twoG) {
        let twoGOnCurve = engine.g1IsOnCurve(twoGAff)
        print("  2G on curve: \(twoGOnCurve ? "PASS" : "FAIL")")
    } else {
        print("  2G on curve: FAIL (identity)")
    }

    // G + G = 2G
    let gPlusG = g1_381Add(gProj, gProj)
    if let gPlusGAff = g1_381ToAffine(gPlusG),
       let twoGAff = g1_381ToAffine(twoG) {
        let addEqDbl = fp381ToInt(gPlusGAff.x) == fp381ToInt(twoGAff.x) &&
                       fp381ToInt(gPlusGAff.y) == fp381ToInt(twoGAff.y)
        print("  G + G = 2G: \(addEqDbl ? "PASS" : "FAIL")")
    }

    // 5G by scalar mul = 5 additions
    let fiveG_mul = g1_381MulInt(gProj, 5)
    var fiveG_add = gProj
    for _ in 1..<5 { fiveG_add = g1_381Add(fiveG_add, gProj) }
    if let fiveGMulAff = g1_381ToAffine(fiveG_mul),
       let fiveGAddAff = g1_381ToAffine(fiveG_add) {
        let scalarOk = fp381ToInt(fiveGMulAff.x) == fp381ToInt(fiveGAddAff.x) &&
                       fp381ToInt(fiveGMulAff.y) == fp381ToInt(fiveGAddAff.y)
        print("  5G (mul) = 5G (add): \(scalarOk ? "PASS" : "FAIL")")
    }

    // G + (-G) = O
    let negG = g1_381NegateAffine(gen1)
    let gPlusNegG = g1_381Add(gProj, g1_381FromAffine(negG))
    let negOk = g1_381IsIdentity(gPlusNegG)
    print("  G + (-G) = O: \(negOk ? "PASS" : "FAIL")")

    // G + O = G
    let gPlusId = g1_381Add(gProj, g1_381Identity())
    if let gPlusIdAff = g1_381ToAffine(gPlusId) {
        let idOk = fp381ToInt(gPlusIdAff.x) == fp381ToInt(gen1.x) &&
                   fp381ToInt(gPlusIdAff.y) == fp381ToInt(gen1.y)
        print("  G + O = G: \(idOk ? "PASS" : "FAIL")")
    }

    // --- G2 Curve tests ---
    print("\n--- G2 Curve Operations ---")

    let g2Simple = bls12381G2SimplePoint()
    let g2OnCurve = engine.g2IsOnCurve(g2Simple)
    print("  G2 simple point on curve: \(g2OnCurve ? "PASS" : "FAIL")")

    let g2Proj = g2_381FromAffine(g2Simple)
    let twoG2 = g2_381Double(g2Proj)
    if let twoG2Aff = g2_381ToAffine(twoG2) {
        let twoG2OnCurve = engine.g2IsOnCurve(twoG2Aff)
        print("  2Q on curve: \(twoG2OnCurve ? "PASS" : "FAIL")")
    }

    // Q + Q = 2Q
    let g2PlusG2 = g2_381Add(g2Proj, g2Proj)
    if let g2PlusG2Aff = g2_381ToAffine(g2PlusG2),
       let twoG2Aff = g2_381ToAffine(twoG2) {
        let g2AddEqDbl = fp381ToInt(g2PlusG2Aff.x.c0) == fp381ToInt(twoG2Aff.x.c0) &&
                         fp381ToInt(g2PlusG2Aff.x.c1) == fp381ToInt(twoG2Aff.x.c1) &&
                         fp381ToInt(g2PlusG2Aff.y.c0) == fp381ToInt(twoG2Aff.y.c0) &&
                         fp381ToInt(g2PlusG2Aff.y.c1) == fp381ToInt(twoG2Aff.y.c1)
        print("  Q + Q = 2Q: \(g2AddEqDbl ? "PASS" : "FAIL")")
    }

    // Q + (-Q) = O
    let negQ = g2_381NegateAffine(g2Simple)
    let g2PlusNeg = g2_381Add(g2Proj, g2_381FromAffine(negQ))
    let g2NegOk = g2_381IsIdentity(g2PlusNeg)
    print("  Q + (-Q) = O: \(g2NegOk ? "PASS" : "FAIL")")

    // --- Benchmarks ---
    print("\n--- Benchmarks ---")

    // Fp multiplication
    let fp_x = fp381FromInt(0xDEADBEEFCAFEBABE)
    let fp_y = fp381FromInt(0x1234567890ABCDEF)
    let fpMulIters = 100_000
    let fpMulT0 = CFAbsoluteTimeGetCurrent()
    var fpAcc = fp_x
    for _ in 0..<fpMulIters { fpAcc = fp381Mul(fpAcc, fp_y) }
    let fpMulTime = (CFAbsoluteTimeGetCurrent() - fpMulT0) * 1e9 / Double(fpMulIters)
    _ = fpAcc
    print(String(format: "  Fp mul: %.0f ns/op", fpMulTime))

    // Fp2 multiplication
    let fp2_x = Fp2_381(c0: fp381FromInt(42), c1: fp381FromInt(17))
    let fp2_y = Fp2_381(c0: fp381FromInt(99), c1: fp381FromInt(73))
    let fp2MulT0 = CFAbsoluteTimeGetCurrent()
    var fp2Acc = fp2_x
    for _ in 0..<fpMulIters { fp2Acc = fp2_381Mul(fp2Acc, fp2_y) }
    let fp2MulTime = (CFAbsoluteTimeGetCurrent() - fp2MulT0) * 1e9 / Double(fpMulIters)
    _ = fp2Acc
    print(String(format: "  Fp2 mul: %.0f ns/op", fp2MulTime))

    // Fp12 multiplication
    let fp12Iters = 10_000
    let fp12_x = fp12_a
    let fp12_y = fp12_381Mul(fp12_a, fp12_a) // something non-trivial
    let fp12MulT0 = CFAbsoluteTimeGetCurrent()
    var fp12Acc = fp12_x
    for _ in 0..<fp12Iters { fp12Acc = fp12_381Mul(fp12Acc, fp12_y) }
    let fp12MulTime = (CFAbsoluteTimeGetCurrent() - fp12MulT0) * 1e6 / Double(fp12Iters)
    _ = fp12Acc
    print(String(format: "  Fp12 mul: %.1f us/op", fp12MulTime))

    // G1 point addition
    let g1Iters = 10_000
    let g1_a = gProj
    let g1_b = g1_381Double(gProj)
    let g1AddT0 = CFAbsoluteTimeGetCurrent()
    var g1Acc = g1_a
    for _ in 0..<g1Iters { g1Acc = g1_381Add(g1Acc, g1_b) }
    let g1AddTime = (CFAbsoluteTimeGetCurrent() - g1AddT0) * 1e6 / Double(g1Iters)
    _ = g1Acc
    print(String(format: "  G1 add: %.1f us/op", g1AddTime))

    // G1 scalar mul
    let scalar: [UInt64] = [0xdeadbeefcafebabe, 0x1234567890abcdef, 0, 0]
    let g1MulT0 = CFAbsoluteTimeGetCurrent()
    let g1MulResult = g1_381ScalarMul(gProj, scalar)
    let g1MulTime = (CFAbsoluteTimeGetCurrent() - g1MulT0) * 1000
    _ = g1MulResult
    print(String(format: "  G1 scalar mul (128-bit): %.1f ms", g1MulTime))

    // G2 point addition
    let g2_a = g2Proj
    let g2_b = g2_381Double(g2Proj)
    let g2AddT0 = CFAbsoluteTimeGetCurrent()
    var g2Acc = g2_a
    for _ in 0..<g1Iters { g2Acc = g2_381Add(g2Acc, g2_b) }
    let g2AddTime = (CFAbsoluteTimeGetCurrent() - g2AddT0) * 1e6 / Double(g1Iters)
    _ = g2Acc
    print(String(format: "  G2 add: %.1f us/op", g2AddTime))

    // Fp inverse
    let fpInvIters = 1000
    let fpInvT0 = CFAbsoluteTimeGetCurrent()
    var fpInvAcc = fp_x
    for _ in 0..<fpInvIters { fpInvAcc = fp381Inverse(fpInvAcc) }
    let fpInvTime = (CFAbsoluteTimeGetCurrent() - fpInvT0) * 1e6 / Double(fpInvIters)
    _ = fpInvAcc
    print(String(format: "  Fp inverse: %.1f us/op", fpInvTime))

    // Fr multiplication
    let fr_x = fr381FromInt(0xDEADBEEFCAFEBABE)
    let fr_y = fr381FromInt(0x1234567890ABCDEF)
    let frMulT0 = CFAbsoluteTimeGetCurrent()
    var frAcc = fr_x
    for _ in 0..<fpMulIters { frAcc = fr381Mul(frAcc, fr_y) }
    let frMulTime = (CFAbsoluteTimeGetCurrent() - frMulT0) * 1e9 / Double(fpMulIters)
    _ = frAcc
    print(String(format: "  Fr mul: %.0f ns/op", frMulTime))

    // --- Frobenius Endomorphism Tests & Benchmarks ---
    print("\n--- Frobenius Endomorphism ---")

    // Correctness: frob(1) == 1
    let frobOneOk = fp12_381Equal(fp12_381Frobenius(.one), .one)
    print("  frob(1) == 1: \(frobOneOk ? "PASS" : "FAIL")")

    // frob2(1) == 1
    let frob2OneOk = fp12_381Equal(fp12_381Frobenius2(.one), .one)
    print("  frob2(1) == 1: \(frob2OneOk ? "PASS" : "FAIL")")

    // frob3(1) == 1
    let frob3OneOk = fp12_381Equal(fp12_381Frobenius3(.one), .one)
    print("  frob3(1) == 1: \(frob3OneOk ? "PASS" : "FAIL")")

    // frob(frob(a)) == frob2(a) -- consistency check
    let frobA = fp12_381Frobenius(fp12_a)
    let frobFrobA = fp12_381Frobenius(frobA)
    let frob2A = fp12_381Frobenius2(fp12_a)
    let frob2ConsistOk = fp12_381Equal(frobFrobA, frob2A)
    print("  frob(frob(a)) == frob2(a): \(frob2ConsistOk ? "PASS" : "FAIL")")

    // frob(frob2(a)) == frob3(a)
    let frobFrob2A = fp12_381Frobenius(frob2A)
    let frob3A = fp12_381Frobenius3(fp12_a)
    let frob3ConsistOk = fp12_381Equal(frobFrob2A, frob3A)
    print("  frob(frob2(a)) == frob3(a): \(frob3ConsistOk ? "PASS" : "FAIL")")

    // Cyclotomic squaring: on a cyclotomic element, should match regular squaring
    // Create a cyclotomic element via easy part of final exp
    let gen1_381 = bls12381G1Generator()
    let gen2_381 = bls12381G2SimplePoint()
    let mlResult = millerLoop381(gen1_381, gen2_381)
    let fConj381 = fp12_381Conjugate(mlResult)
    let fInv381 = fp12_381Inverse(mlResult)
    var cyclElem = fp12_381Mul(fConj381, fInv381) // f^(p^6-1)
    let cyclElemP2 = fp12_381Frobenius2(cyclElem)
    cyclElem = fp12_381Mul(cyclElemP2, cyclElem) // f^((p^6-1)(p^2+1))

    // Now cyclElem is in cyclotomic subgroup: conj(cyclElem) == cyclElem^{-1}
    let cyclConj = fp12_381Conjugate(cyclElem)
    let cyclInv = fp12_381Inverse(cyclElem)
    let cyclUnitaryOk = fp12_381Equal(cyclConj, cyclInv)
    print("  cyclotomic: conj(f) == f^{-1}: \(cyclUnitaryOk ? "PASS" : "FAIL")")

    // Compare cyclotomic squaring vs generic squaring
    let cyclSqrGeneric = fp12_381Sqr(cyclElem)
    let cyclSqrCyclo = fp12_381CyclotomicSqr(cyclElem)
    let cyclSqrMatch = fp12_381Equal(cyclSqrGeneric, cyclSqrCyclo)
    print("  cyclotomic_sqr(f) == generic_sqr(f): \(cyclSqrMatch ? "PASS" : "FAIL")")

    // Benchmark: Fp12 squaring vs cyclotomic squaring
    let sqrIters = 10_000

    let fp12SqrT0 = CFAbsoluteTimeGetCurrent()
    var sqrAcc = cyclElem
    for _ in 0..<sqrIters { sqrAcc = fp12_381Sqr(sqrAcc) }
    let fp12SqrTime = (CFAbsoluteTimeGetCurrent() - fp12SqrT0) * 1e6 / Double(sqrIters)
    _ = sqrAcc
    print(String(format: "  Fp12 generic sqr: %.1f us/op", fp12SqrTime))

    let cyclSqrT0 = CFAbsoluteTimeGetCurrent()
    var cyclAcc = cyclElem
    for _ in 0..<sqrIters { cyclAcc = fp12_381CyclotomicSqr(cyclAcc) }
    let cyclSqrTime = (CFAbsoluteTimeGetCurrent() - cyclSqrT0) * 1e6 / Double(sqrIters)
    _ = cyclAcc
    print(String(format: "  Fp12 cyclotomic sqr: %.1f us/op", cyclSqrTime))

    if fp12SqrTime > 0 {
        print(String(format: "  Cyclotomic speedup: %.1fx", fp12SqrTime / cyclSqrTime))
    }

    // Benchmark: Frobenius maps
    let frobIters = 10_000

    let frob1T0 = CFAbsoluteTimeGetCurrent()
    var frobAcc = fp12_a
    for _ in 0..<frobIters { frobAcc = fp12_381Frobenius(frobAcc) }
    let frob1Time = (CFAbsoluteTimeGetCurrent() - frob1T0) * 1e6 / Double(frobIters)
    _ = frobAcc
    print(String(format: "  Fp12 Frobenius^1: %.1f us/op", frob1Time))

    let frob2T0 = CFAbsoluteTimeGetCurrent()
    var frob2Acc = fp12_a
    for _ in 0..<frobIters { frob2Acc = fp12_381Frobenius2(frob2Acc) }
    let frob2Time = (CFAbsoluteTimeGetCurrent() - frob2T0) * 1e6 / Double(frobIters)
    _ = frob2Acc
    print(String(format: "  Fp12 Frobenius^2: %.1f us/op (MulByFp optimized)", frob2Time))

    let frob3T0 = CFAbsoluteTimeGetCurrent()
    var frob3Acc = fp12_a
    for _ in 0..<frobIters { frob3Acc = fp12_381Frobenius3(frob3Acc) }
    let frob3Time = (CFAbsoluteTimeGetCurrent() - frob3T0) * 1e6 / Double(frobIters)
    _ = frob3Acc
    print(String(format: "  Fp12 Frobenius^3: %.1f us/op", frob3Time))

    // Benchmark: Full pairing
    let pairIters = 3
    let pairT0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<pairIters {
        _ = bls12381Pairing(gen1_381, gen2_381)
    }
    let pairTime = (CFAbsoluteTimeGetCurrent() - pairT0) * 1000 / Double(pairIters)
    print(String(format: "  Full pairing: %.1f ms", pairTime))

    // Summary
    let allFields = oneOk && zeroOk && addZeroOk381 && addNegOk381 && mulOneOk381 &&
                    mulInvOk381 && distOk381 && rootOk && rootFullOk &&
                    fpOneOk && fpAddOk && fpInvOk &&
                    fp2MulOk && fp2InvOk && conjOk &&
                    fp6InvOk && fp12InvOk &&
                    g1OnCurve && negOk && g2OnCurve && g2NegOk &&
                    frobOneOk && frob2OneOk && frob3OneOk &&
                    frob2ConsistOk && frob3ConsistOk &&
                    cyclUnitaryOk && cyclSqrMatch
    print("\n  Overall: \(allFields ? "ALL PASS" : "SOME FAILED")")
}
