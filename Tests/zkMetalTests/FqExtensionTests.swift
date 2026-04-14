// BLS12-377 Fq2/Fq6/Fq12 extension field tests
// Verifies Karatsuba multiplication and basic field arithmetic

import zkMetal

func runFqExtensionTests() {
    suite("BLS12-377 Fq2 arithmetic")
    testFq2Basic()
    testFq2Karatsuba()

    suite("BLS12-377 Fq6 arithmetic")
    testFq6Basic()

    suite("BLS12-377 Fq12 arithmetic")
    testFq12Basic()
}

// Helper: compare two Fq377 values via their integer representation
private func fqEq(_ a: Fq377, _ b: Fq377) -> Bool {
    let al = fq377ToInt(a)
    let bl = fq377ToInt(b)
    return al[0] == bl[0] && al[1] == bl[1] && al[2] == bl[2] &&
           al[3] == bl[3] && al[4] == bl[4] && al[5] == bl[5]
}

// Helper: compare two Fq2_377 values
private func fq2Eq(_ a: Fq2_377, _ b: Fq2_377) -> Bool {
    fqEq(a.c0, b.c0) && fqEq(a.c1, b.c1)
}

// Helper: compare two Fq6_377 values
private func fq6Eq(_ a: Fq6_377, _ b: Fq6_377) -> Bool {
    fq2Eq(a.c0, b.c0) && fq2Eq(a.c1, b.c1) && fq2Eq(a.c2, b.c2)
}

// Helper: compare two Fq12_377 values
private func fq12Eq(_ a: Fq12_377, _ b: Fq12_377) -> Bool {
    fq6Eq(a.c0, b.c0) && fq6Eq(a.c1, b.c1)
}

// Helper: Fq377 from small integer
private func fq(_ val: UInt64) -> Fq377 {
    fq377FromInt(val)
}

// Helper: check Fq377 is zero
private func fqIsZero(_ a: Fq377) -> Bool {
    a.isZero
}

private func testFq2Basic() {
    // Zero and identity
    expect(Fq2_377.zero.isZero, "Fq2 zero check")
    expect(!Fq2_377.one.isZero, "Fq2 one is non-zero")
    expect(fqEq(Fq2_377.one.c0, Fq377.one), "Fq2 one c0 = Fq377.one")
    expect(fqIsZero(Fq2_377.one.c1), "Fq2 one c1 = 0")

    // Addition: x + 0 = x
    let a = Fq2_377(c0: fq(3), c1: fq(5))
    let sum = fq2_377Add(a, .zero)
    expect(fqEq(sum.c0, a.c0) && fqEq(sum.c1, a.c1), "Fq2 add zero")

    // Subtraction: x - 0 = x
    let diff = fq2_377Sub(a, .zero)
    expect(fqEq(diff.c0, a.c0) && fqEq(diff.c1, a.c1), "Fq2 sub zero")

    // Self-inverse: x - x = 0
    let negNeg = fq2_377Sub(a, a)
    expect(negNeg.isZero, "Fq2 x - x = 0")

    // Negation: x + (-x) = 0
    let neg = fq2_377Neg(a)
    let sumNeg = fq2_377Add(a, neg)
    expect(sumNeg.isZero, "Fq2 x + (-x) = 0")

    // Doubling: 2x = x + x
    let doubled = fq2_377Double(a)
    let added = fq2_377Add(a, a)
    expect(fqEq(doubled.c0, added.c0) && fqEq(doubled.c1, added.c1), "Fq2 doubling")

    // Multiplication by 1
    let mul1 = fq2_377Mul(a, .one)
    expect(fqEq(mul1.c0, a.c0) && fqEq(mul1.c1, a.c1), "Fq2 mul by one")

    // Multiplication by 0
    let mul0 = fq2_377Mul(a, .zero)
    expect(mul0.isZero, "Fq2 mul by zero")

    // Inverse: x * x^{-1} = 1
    let inv = fq2_377Inverse(a)
    let prod = fq2_377Mul(a, inv)
    expect(fqEq(prod.c0, Fq377.one) && fqIsZero(prod.c1), "Fq2 inverse")

    // Squaring: x^2 = x * x
    let sqr = fq2_377Sqr(a)
    let mul = fq2_377Mul(a, a)
    expect(fqEq(sqr.c0, mul.c0) && fqEq(sqr.c1, mul.c1), "Fq2 squaring")

    // Conjugation: (a + bi)^{-1} = (a - bi) / (a^2 + b^2)
    let conj = fq2_377Conjugate(a)
    expect(fqEq(conj.c0, a.c0) && fqEq(conj.c1, fq377Neg(a.c1)), "Fq2 conjugate")

    // Non-residue multiplication: (a+bi)(1+i) = (a-b) + (a+b)i
    let nr = fq2_377MulByNonResidue(a)
    let expectedNrC0 = fq377Sub(a.c0, a.c1)
    let expectedNrC1 = fq377Add(a.c0, a.c1)
    expect(fqEq(nr.c0, expectedNrC0) && fqEq(nr.c1, expectedNrC1), "Fq2 mul by non-residue")
}

private func testFq2Karatsuba() {
    // Verify Karatsuba consistency: (a+bi)(c+di) = (ac-bd) + (ad+bc-bd)i
    let a = Fq2_377(c0: fq(7), c1: fq(11))
    let b = Fq2_377(c0: fq(13), c1: fq(17))

    let karatsuba = fq2_377Mul(a, b)

    // Manual schoolbook multiplication
    let ac = fq377Mul(a.c0, b.c0)
    let bd = fq377Mul(a.c1, b.c1)
    let ad = fq377Mul(a.c0, b.c1)
    let bc = fq377Mul(a.c1, b.c0)

    let expectedC0 = fq377Sub(ac, bd)      // ac - bd
    let expectedC1 = fq377Add(ad, bc)     // ad + bc

    expect(fqEq(karatsuba.c0, expectedC0) && fqEq(karatsuba.c1, expectedC1), "Fq2 Karatsuba correctness")

    // Verify (a+b)^2 formula for squaring: (2a)^2 should equal 4 * a^2
    let apb = fq2_377Add(a, a)  // 2a
    let sqrApb = fq2_377Sqr(apb)

    // (2a)^2 = 4a^2
    let aSqr = fq2_377Sqr(a)
    let fourASqr = Fq2_377(c0: fq377Mul(aSqr.c0, fq(4)), c1: fq377Mul(aSqr.c1, fq(4)))
    expect(fqEq(sqrApb.c0, fourASqr.c0) && fqEq(sqrApb.c1, fourASqr.c1), "Fq2 Karatsuba squaring")
}

private func testFq6Basic() {
    // Zero and identity
    expect(Fq6_377.zero.isZero, "Fq6 zero check")
    expect(!Fq6_377.one.isZero, "Fq6 one is non-zero")
    expect(fqEq(Fq6_377.one.c0.c0, Fq377.one), "Fq6 one c0.c0 = Fq377.one")
    expect(fqIsZero(Fq6_377.one.c0.c1), "Fq6 one c0.c1 = 0")
    expect(Fq6_377.one.c1.isZero, "Fq6 one c1 = 0")
    expect(Fq6_377.one.c2.isZero, "Fq6 one c2 = 0")

    // Addition: x + 0 = x
    let a = Fq6_377(
        c0: Fq2_377(c0: fq(3), c1: fq(5)),
        c1: Fq2_377(c0: fq(7), c1: fq(11)),
        c2: Fq2_377(c0: fq(13), c1: fq(17)))
    let sum = fq6_377Add(a, .zero)
    expect(fqEq(sum.c0.c0, a.c0.c0) && fqEq(sum.c1.c0, a.c1.c0) && fqEq(sum.c2.c0, a.c2.c0),
           "Fq6 add zero")

    // Subtraction: x - 0 = x
    let diff = fq6_377Sub(a, .zero)
    expect(fqEq(diff.c0.c0, a.c0.c0) && fqEq(diff.c1.c0, a.c1.c0) && fqEq(diff.c2.c0, a.c2.c0),
           "Fq6 sub zero")

    // Self-inverse: x - x = 0
    let negNeg = fq6_377Sub(a, a)
    expect(negNeg.isZero, "Fq6 x - x = 0")

    // Negation: x + (-x) = 0
    let neg = fq6_377Neg(a)
    let sumNeg = fq6_377Add(a, neg)
    expect(sumNeg.isZero, "Fq6 x + (-x) = 0")

    // Multiplication by 1
    let mul1 = fq6_377Mul(a, .one)
    expect(fqEq(mul1.c0.c0, a.c0.c0) && fqEq(mul1.c1.c0, a.c1.c0) && fqEq(mul1.c2.c0, a.c2.c0),
           "Fq6 mul by one")

    // Multiplication by 0
    let mul0 = fq6_377Mul(a, .zero)
    expect(mul0.isZero, "Fq6 mul by zero")

    // Inverse: x * x^{-1} = 1
    let inv = fq6_377Inverse(a)
    let prod = fq6_377Mul(a, inv)
    expect(fqEq(prod.c0.c0, Fq377.one) && fqIsZero(prod.c0.c1) &&
           prod.c1.isZero && prod.c2.isZero, "Fq6 inverse")

    // Squaring: x^2 = x * x
    let sqr = fq6_377Sqr(a)
    let mul = fq6_377Mul(a, a)
    expect(fqEq(sqr.c0.c0, mul.c0.c0) && fqEq(sqr.c1.c0, mul.c1.c0) && fqEq(sqr.c2.c0, mul.c2.c0),
           "Fq6 squaring")

    // Multiply by v: v*(a0 + a1*v + a2*v^2) = a2*ξ + a0*v + a1*v^2
    let mulByV = fq6_377MulByV(a)
    let expectedC0 = fq2_377Mul(a.c2, Fq2_377(c0: .one, c1: .one))  // a2 * ξ
    expect(fq2Eq(mulByV.c0, expectedC0), "Fq6 mul by v c0")
    expect(fq2Eq(mulByV.c1, a.c0), "Fq6 mul by v c1")
    expect(fq2Eq(mulByV.c2, a.c1), "Fq6 mul by v c2")
}

private func testFq12Basic() {
    // Zero and identity
    expect(Fq12_377.zero.isZero, "Fq12 zero check")
    expect(!Fq12_377.one.isZero, "Fq12 one is non-zero")
    expect(fqEq(Fq12_377.one.c0.c0.c0, Fq377.one), "Fq12 one structure")

    // Addition: x + 0 = x
    let a = Fq12_377(
        c0: Fq6_377(
            c0: Fq2_377(c0: fq(3), c1: fq(5)),
            c1: Fq2_377(c0: fq(7), c1: fq(11)),
            c2: Fq2_377(c0: fq(13), c1: fq(17))),
        c1: Fq6_377(
            c0: Fq2_377(c0: fq(19), c1: fq(23)),
            c1: Fq2_377(c0: fq(29), c1: fq(31)),
            c2: Fq2_377(c0: fq(37), c1: fq(41))))
    let sum = fq12_377Add(a, .zero)
    expect(fq12Eq(sum, a), "Fq12 add zero")

    // Subtraction: x - 0 = x
    let diff = fq12_377Sub(a, .zero)
    expect(fq12Eq(diff, a), "Fq12 sub zero")

    // Self-inverse: x - x = 0
    let negNeg = fq12_377Sub(a, a)
    expect(negNeg.isZero, "Fq12 x - x = 0")

    // Negation: x + (-x) = 0
    let neg = fq12_377Neg(a)
    let sumNeg = fq12_377Add(a, neg)
    expect(sumNeg.isZero, "Fq12 x + (-x) = 0")

    // Multiplication by 1
    let mul1 = fq12_377Mul(a, .one)
    expect(fq12Eq(mul1, a), "Fq12 mul by one")

    // Multiplication by 0
    let mul0 = fq12_377Mul(a, .zero)
    expect(mul0.isZero, "Fq12 mul by zero")

    // Inverse: x * x^{-1} = 1
    let inv = fq12_377Inverse(a)
    let prod = fq12_377Mul(a, inv)
    expect(fqEq(prod.c0.c0.c0, Fq377.one) && fqIsZero(prod.c0.c0.c1) &&
           prod.c0.c1.isZero && prod.c0.c2.isZero &&
           prod.c1.isZero, "Fq12 inverse")

    // Squaring: x^2 = x * x
    let sqr = fq12_377Sqr(a)
    let mul = fq12_377Mul(a, a)
    expect(fq12Eq(sqr, mul), "Fq12 squaring")

    // Conjugation: negates c1
    let conj = fq12_377Conjugate(a)
    expect(fqEq(conj.c0.c0.c0, a.c0.c0.c0) &&
           fq2Eq(conj.c1.c0, fq2_377Neg(a.c1.c0)),
           "Fq12 conjugate")
}
