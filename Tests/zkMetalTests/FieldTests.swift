import zkMetal

func runFieldTests() {
    suite("BN254 Fp Field")
    let a = fpFromInt(42)
    expect(fpToInt(a)[0] == 42, "Montgomery round-trip")
    expect(fpToInt(fpAdd(fpFromInt(42), fpFromInt(100)))[0] == 142, "Addition")
    expect(fpToInt(fpMul(fpFromInt(42), fpFromInt(100)))[0] == 4200, "Multiplication")
    expect(fpToInt(fpSub(fpFromInt(100), fpFromInt(42)))[0] == 58, "Subtraction")
    let aInv = fpInverse(a)
    expect(fpToInt(fpMul(a, aInv))[0] == 1, "Inverse")

    suite("BN254 Fr Field")
    expect(frToInt(frFromInt(42))[0] == 42, "Fr round-trip")
    expect(frToInt(frMul(frFromInt(7), frFromInt(6)))[0] == 42, "Fr 7*6=42")
    expect(frEqual(frMul(frFromInt(42), frInverse(frFromInt(42))), Fr.one), "Fr inverse")

    suite("secp256k1 Field")
    expect(secpFrToInt(secpFrFromInt(1))[0] == 1, "Fr round-trip")
    expect(secpFrToInt(secpFrMul(secpFrFromInt(42), secpFrFromInt(7)))[0] == 294, "Fr 42*7=294")
    expect(secpFrMul(secpFrFromInt(42), secpFrInverse(secpFrFromInt(42))) == SecpFr.one, "Fr inverse")

    let elems = [secpFrFromInt(3), secpFrFromInt(7), secpFrFromInt(42), secpFrFromInt(100)]
    let invs = secpFrBatchInverse(elems)
    for i in 0..<elems.count {
        expect(secpFrMul(elems[i], invs[i]) == SecpFr.one, "Batch inv[\(i)]")
    }

    if let sq = secpSqrt(secpFromInt(4)) {
        expect(secpToInt(sq)[0] == 2, "sqrt(4)=2")
    } else { expect(false, "sqrt(4) should exist") }
    expect(secpSqrt(secpFromInt(3)) == nil, "sqrt(3) non-residue")

    suite("Mersenne31 Field")
    expect(m31Add(M31(v: 42), M31(v: 100)).v == 142, "M31 add")
    expect(m31Mul(M31(v: 42), M31(v: 100)).v == 4200, "M31 mul")
    expect(m31Sub(M31(v: 100), M31(v: 42)).v == 58, "M31 sub")
    expect(m31Mul(M31(v: 42), m31Inverse(M31(v: 42))).v == 1, "M31 inverse")
    expect(m31Add(M31(v: M31.P - 1), M31.one).v == 0, "M31 wraparound")
    expect(m31Add(M31(v: 42), m31Neg(M31(v: 42))).v == 0, "M31 negation")

    let c = CM31(a: M31(v: 3), b: M31(v: 4))
    let cOne = cm31Mul(c, cm31Inverse(c))
    expect(cOne.a.v == 1, "CM31 inv real")
    expect(cOne.b.v == 0, "CM31 inv imag")

    var g = CirclePoint.generator
    expect(g.isOnCircle, "Generator on circle")
    for _ in 0..<31 { g = circleGroupMul(g, g) }
    expect(g == CirclePoint.identity, "Circle gen order 2^31")
}
