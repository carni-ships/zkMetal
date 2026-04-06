import zkMetal
import Foundation

public func runGPUBabyBearExtensionTests() {
    suite("Bb4 Field Axioms")
    testBb4FieldAxioms()

    suite("Bb4 Additive Group")
    testBb4AdditiveGroup()

    suite("Bb4 Multiplication")
    testBb4Multiplication()

    suite("Bb4 Squaring")
    testBb4Squaring()

    suite("Bb4 Inverse")
    testBb4Inverse()

    suite("Bb4 Scalar Mul & MulByX")
    testBb4ScalarAndShift()

    suite("Bb4 Frobenius Endomorphism")
    testBb4Frobenius()

    suite("Bb4 Base Field Conversion")
    testBb4Conversion()

    suite("Bb4 Trace Map")
    testBb4Trace()

    suite("Bb4 Random Generation")
    testBb4Random()

    suite("Bb4 Batch Add/Sub/Mul")
    testBb4BatchArithmetic()

    suite("Bb4 Batch Inverse")
    testBb4BatchInverse()

    suite("Bb4 Inner Product")
    testBb4InnerProduct()

    suite("Bb4 Extension NTT")
    testBb4NTT()

    suite("Bb4 Polynomial Multiply via NTT")
    testBb4PolyMul()

    suite("Bb4 GPU Engine")
    testBb4Engine()

    suite("Bb4 Performance (10K muls)")
    benchmarkBb4Mul()
}

// MARK: - Field Axioms

private func testBb4FieldAxioms() {
    let a = Bb4(c0: Bb(v: 42), c1: Bb(v: 17), c2: Bb(v: 99), c3: Bb(v: 7))
    let b = Bb4(c0: Bb(v: 13), c1: Bb(v: 55), c2: Bb(v: 3), c3: Bb(v: 200))
    let c = Bb4(c0: Bb(v: 1000), c1: Bb(v: 2000), c2: Bb(v: 3000), c3: Bb(v: 4000))

    // Commutativity of addition
    expect(bb4Add(a, b) == bb4Add(b, a), "Bb4 add commutative")

    // Commutativity of multiplication
    expect(bb4Mul(a, b) == bb4Mul(b, a), "Bb4 mul commutative")

    // Associativity of addition
    expect(bb4Add(bb4Add(a, b), c) == bb4Add(a, bb4Add(b, c)), "Bb4 add associative")

    // Associativity of multiplication
    expect(bb4Mul(bb4Mul(a, b), c) == bb4Mul(a, bb4Mul(b, c)), "Bb4 mul associative")

    // Distributivity: a * (b + c) = a*b + a*c
    expect(bb4Mul(a, bb4Add(b, c)) == bb4Add(bb4Mul(a, b), bb4Mul(a, c)),
           "Bb4 distributive")

    // Additive identity
    expect(bb4Add(a, Bb4.zero) == a, "Bb4 add identity")

    // Multiplicative identity
    expect(bb4Mul(a, Bb4.one) == a, "Bb4 mul identity")

    // Zero absorbs multiplication
    expect(bb4Mul(a, Bb4.zero) == Bb4.zero, "Bb4 mul by zero")
}

// MARK: - Additive Group

private func testBb4AdditiveGroup() {
    let a = Bb4(c0: Bb(v: 42), c1: Bb(v: 17), c2: Bb(v: 99), c3: Bb(v: 7))
    let b = Bb4(c0: Bb(v: 13), c1: Bb(v: 55), c2: Bb(v: 3), c3: Bb(v: 200))

    // a + (-a) = 0
    expect(bb4Add(a, bb4Neg(a)) == Bb4.zero, "Bb4 a + (-a) = 0")

    // a - b + b = a
    expect(bb4Add(bb4Sub(a, b), b) == a, "Bb4 sub then add restores original")

    // Double = add self
    expect(bb4Double(a) == bb4Add(a, a), "Bb4 double = add self")

    // Subtraction anti-commutativity: a - b = -(b - a)
    expect(bb4Sub(a, b) == bb4Neg(bb4Sub(b, a)), "Bb4 sub anti-commutative")
}

// MARK: - Multiplication

private func testBb4Multiplication() {
    let a = Bb4(c0: Bb(v: 42), c1: Bb(v: 17), c2: Bb(v: 99), c3: Bb(v: 7))

    // a * 1 = a
    expect(bb4Mul(a, Bb4.one) == a, "Bb4 a * 1 = a")

    // a * 0 = 0
    expect(bb4Mul(a, Bb4.zero) == Bb4.zero, "Bb4 a * 0 = 0")

    // Pure x element: (0, 1, 0, 0) * (0, 1, 0, 0) = (0, 0, 1, 0) (x * x = x^2)
    let x = Bb4(c0: Bb.zero, c1: Bb.one, c2: Bb.zero, c3: Bb.zero)
    let x2 = bb4Mul(x, x)
    expect(x2 == Bb4(c0: Bb.zero, c1: Bb.zero, c2: Bb.one, c3: Bb.zero), "Bb4 x*x = x^2")

    // x^2 * x^2 = x^4 = W = 11
    let x4 = bb4Mul(x2, x2)
    expect(x4 == Bb4(c0: Bb(v: 11), c1: Bb.zero, c2: Bb.zero, c3: Bb.zero),
           "Bb4 x^4 = 11")

    // Verify x^4 = W via repeated multiplication
    let x3 = bb4Mul(x2, x)
    let x4b = bb4Mul(x3, x)
    expect(x4b == bb4FromBase(Bb(v: 11)), "Bb4 x * x * x * x = 11")

    // Base field elements multiply as in Bb
    let ba = Bb4(from: Bb(v: 7))
    let bb = Bb4(from: Bb(v: 13))
    let prod = bb4Mul(ba, bb)
    expect(prod == Bb4(from: bbMul(Bb(v: 7), Bb(v: 13))), "Bb4 base mul consistent")
}

// MARK: - Squaring

private func testBb4Squaring() {
    let testCases: [Bb4] = [
        Bb4(c0: Bb(v: 42), c1: Bb(v: 17), c2: Bb(v: 99), c3: Bb(v: 7)),
        Bb4(c0: Bb(v: 1000000), c1: Bb(v: 2000000), c2: Bb(v: 0), c3: Bb(v: 500)),
        Bb4(c0: Bb.zero, c1: Bb.one, c2: Bb.zero, c3: Bb.zero),  // x
        Bb4(c0: Bb.one, c1: Bb.one, c2: Bb.one, c3: Bb.one),     // 1 + x + x^2 + x^3
        Bb4(c0: Bb(v: Bb.P - 1), c1: Bb(v: Bb.P - 1), c2: Bb(v: Bb.P - 1), c3: Bb(v: Bb.P - 1)),
    ]

    for (i, a) in testCases.enumerated() {
        expect(bb4Sqr(a) == bb4Mul(a, a), "Bb4 sqr[\(i)] = mul self")
    }
}

// MARK: - Inverse

private func testBb4Inverse() {
    // inv(1) = 1
    let oneInv = bb4Inverse(Bb4.one)
    expect(oneInv == Bb4.one, "Bb4 inv(1) = 1")

    // a * inv(a) = 1 for several values
    let testValues: [Bb4] = [
        Bb4(c0: Bb(v: 42), c1: Bb(v: 17), c2: Bb(v: 99), c3: Bb(v: 7)),
        Bb4(c0: Bb(v: 1), c1: Bb.zero, c2: Bb.zero, c3: Bb.zero),  // base field
        Bb4(c0: Bb.zero, c1: Bb.one, c2: Bb.zero, c3: Bb.zero),    // pure x
        Bb4(c0: Bb(v: 1000000), c1: Bb(v: 2000000), c2: Bb(v: 500), c3: Bb(v: 777)),
        Bb4(c0: Bb(v: Bb.P - 1), c1: Bb(v: 1), c2: Bb(v: Bb.P - 2), c3: Bb(v: 3)),
    ]

    for (i, val) in testValues.enumerated() {
        let inv = bb4Inverse(val)
        let product = bb4Mul(val, inv)
        expect(product == Bb4.one, "Bb4 a * inv(a) = 1 [\(i)]")
    }

    // Double inverse: inv(inv(a)) = a
    let a = Bb4(c0: Bb(v: 42), c1: Bb(v: 17), c2: Bb(v: 99), c3: Bb(v: 7))
    let invInv = bb4Inverse(bb4Inverse(a))
    expect(invInv == a, "Bb4 inv(inv(a)) = a")

    // inv(a*b) = inv(a) * inv(b)
    let b = Bb4(c0: Bb(v: 13), c1: Bb(v: 55), c2: Bb(v: 3), c3: Bb(v: 200))
    let invProd = bb4Inverse(bb4Mul(a, b))
    let prodInv = bb4Mul(bb4Inverse(a), bb4Inverse(b))
    expect(invProd == prodInv, "Bb4 inv(a*b) = inv(a)*inv(b)")
}

// MARK: - Scalar Mul & MulByX

private func testBb4ScalarAndShift() {
    let a = Bb4(c0: Bb(v: 42), c1: Bb(v: 17), c2: Bb(v: 99), c3: Bb(v: 7))
    let s = Bb(v: 5)

    // Scalar mul = multiplying each component
    let scaled = bb4ScalarMul(s, a)
    expect(scaled.c0 == bbMul(s, a.c0), "Bb4 scalar mul c0")
    expect(scaled.c1 == bbMul(s, a.c1), "Bb4 scalar mul c1")
    expect(scaled.c2 == bbMul(s, a.c2), "Bb4 scalar mul c2")
    expect(scaled.c3 == bbMul(s, a.c3), "Bb4 scalar mul c3")

    // Scalar mul by 1 = identity
    expect(bb4ScalarMul(Bb.one, a) == a, "Bb4 scalar mul by 1")

    // Scalar mul by 0 = zero
    expect(bb4ScalarMul(Bb.zero, a) == Bb4.zero, "Bb4 scalar mul by 0")

    // MulByX consistency: mulByX(a) = x * a
    let x = Bb4(c0: Bb.zero, c1: Bb.one, c2: Bb.zero, c3: Bb.zero)
    expect(bb4MulByX(a) == bb4Mul(x, a), "Bb4 mulByX = x * a")
}

// MARK: - Frobenius

private func testBb4Frobenius() {
    let a = Bb4(c0: Bb(v: 42), c1: Bb(v: 17), c2: Bb(v: 99), c3: Bb(v: 7))

    // Frobenius is a field homomorphism: phi(a*b) = phi(a)*phi(b)
    let b = Bb4(c0: Bb(v: 13), c1: Bb(v: 55), c2: Bb(v: 3), c3: Bb(v: 200))
    let phiAB = bb4Frobenius(bb4Mul(a, b))
    let phiA_phiB = bb4Mul(bb4Frobenius(a), bb4Frobenius(b))
    expect(phiAB == phiA_phiB, "Bb4 Frobenius multiplicative")

    // Frobenius preserves addition: phi(a+b) = phi(a)+phi(b)
    let phiAplusB = bb4Frobenius(bb4Add(a, b))
    let phiA_plusPhiB = bb4Add(bb4Frobenius(a), bb4Frobenius(b))
    expect(phiAplusB == phiA_plusPhiB, "Bb4 Frobenius additive")

    // phi fixes base field: phi(c) = c for c in Bb
    let base = Bb4(from: Bb(v: 42))
    expect(bb4Frobenius(base) == base, "Bb4 Frobenius fixes base field")

    // phi^4 = identity (order divides 4 for Bb4/Bb)
    expect(bb4FrobeniusK(a, 4) == a, "Bb4 Frobenius^4 = identity")

    // phi(1) = 1
    expect(bb4Frobenius(Bb4.one) == Bb4.one, "Bb4 Frobenius(1) = 1")
}

// MARK: - Conversion

private func testBb4Conversion() {
    // Embed and extract round-trip
    let x = Bb(v: 42)
    let ext = bb4FromBase(x)
    expect(ext.isInBaseField, "Bb4 from base is in base field")
    expect(bb4ToBase(ext) == x, "Bb4 to/from base round-trip")

    // Zero round-trip
    let zExt = bb4FromBase(Bb.zero)
    expect(zExt == Bb4.zero, "Bb4 from base zero = Bb4 zero")

    // One round-trip
    let oExt = bb4FromBase(Bb.one)
    expect(oExt == Bb4.one, "Bb4 from base one = Bb4 one")

    // Base field arithmetic is consistent through embedding
    let a = Bb(v: 7)
    let b = Bb(v: 13)
    let sumBase = bbAdd(a, b)
    let sumExt = bb4Add(bb4FromBase(a), bb4FromBase(b))
    expect(sumExt == bb4FromBase(sumBase), "Bb4 embedding preserves add")

    let prodBase = bbMul(a, b)
    let prodExt = bb4Mul(bb4FromBase(a), bb4FromBase(b))
    expect(prodExt == bb4FromBase(prodBase), "Bb4 embedding preserves mul")
}

// MARK: - Trace

private func testBb4Trace() {
    // Trace of base field element c is 4*c (since phi fixes Bb, so Tr = 4c)
    let c = Bb(v: 10)
    let tr = bb4Trace(bb4FromBase(c))
    let expected = bbMul(Bb(v: 4), c)
    expect(tr == expected, "Bb4 trace of base element = 4*c")

    // Trace is Bb-linear: Tr(a + b) = Tr(a) + Tr(b)
    let a = Bb4(c0: Bb(v: 42), c1: Bb(v: 17), c2: Bb(v: 99), c3: Bb(v: 7))
    let b = Bb4(c0: Bb(v: 13), c1: Bb(v: 55), c2: Bb(v: 3), c3: Bb(v: 200))
    let trSum = bb4Trace(bb4Add(a, b))
    let sumTr = bbAdd(bb4Trace(a), bb4Trace(b))
    expect(trSum == sumTr, "Bb4 trace is additive")

    // Trace of scalar multiple: Tr(s*a) = s*Tr(a)
    let s = Bb(v: 5)
    let trScaled = bb4Trace(bb4ScalarMul(s, a))
    let scaledTr = bbMul(s, bb4Trace(a))
    expect(trScaled == scaledTr, "Bb4 trace is Bb-linear")
}

// MARK: - Random

private func testBb4Random() {
    var seed: UInt64 = 0xDEADBEEF

    // Generate elements and verify they are valid (all components < p)
    let elems = bb4RandomArray(count: 100, seed: &seed)
    expect(elems.count == 100, "Bb4 random array count")

    var allValid = true
    for e in elems {
        if e.c0.v >= Bb.P || e.c1.v >= Bb.P || e.c2.v >= Bb.P || e.c3.v >= Bb.P {
            allValid = false
            break
        }
    }
    expect(allValid, "Bb4 random elements all < p")

    // Different seeds give different results
    var seed2: UInt64 = 0xCAFEBABE
    let elems2 = bb4RandomArray(count: 10, seed: &seed2)
    var seed3: UInt64 = 0xDEADBEEF
    let elems3 = bb4RandomArray(count: 10, seed: &seed3)
    // Same seed should give same result
    var seed4: UInt64 = 0xDEADBEEF
    let elems4 = bb4RandomArray(count: 10, seed: &seed4)
    expect(elems3[0] == elems4[0], "Bb4 same seed = same elements")
    expect(elems2[0] != elems3[0], "Bb4 different seed = different elements")

    // Not all zero
    var hasNonZero = false
    for e in elems {
        if !e.isZero { hasNonZero = true; break }
    }
    expect(hasNonZero, "Bb4 random elements not all zero")
}

// MARK: - Batch Arithmetic

private func testBb4BatchArithmetic() {
    let n = 16
    var seed: UInt64 = 0x12345678
    let xs = bb4RandomArray(count: n, seed: &seed)
    let ys = bb4RandomArray(count: n, seed: &seed)

    // Batch add
    let batchSum = bb4BatchAdd(xs, ys)
    for i in 0..<n {
        expect(batchSum[i] == bb4Add(xs[i], ys[i]), "Bb4 batch add[\(i)]")
    }

    // Batch sub
    let batchDiff = bb4BatchSub(xs, ys)
    for i in 0..<n {
        expect(batchDiff[i] == bb4Sub(xs[i], ys[i]), "Bb4 batch sub[\(i)]")
    }

    // Batch mul
    let batchProd = bb4BatchMul(xs, ys)
    for i in 0..<n {
        expect(batchProd[i] == bb4Mul(xs[i], ys[i]), "Bb4 batch mul[\(i)]")
    }

    // Batch sqr
    let batchSqr = bb4BatchSqr(xs)
    for i in 0..<n {
        expect(batchSqr[i] == bb4Sqr(xs[i]), "Bb4 batch sqr[\(i)]")
    }

    // Batch scalar mul
    let s = Bb(v: 42)
    let batchScalar = bb4BatchScalarMul(s, xs)
    for i in 0..<n {
        expect(batchScalar[i] == bb4ScalarMul(s, xs[i]), "Bb4 batch scalar mul[\(i)]")
    }
}

// MARK: - Batch Inverse

private func testBb4BatchInverse() {
    let n = 32
    var seed: UInt64 = 0xFEEDFACE

    // Generate nonzero elements
    var arr = [Bb4]()
    for _ in 0..<n {
        var elem = bb4Random(seed: &seed)
        // Ensure nonzero
        if elem.isZero { elem.c0 = Bb.one }
        arr.append(elem)
    }

    let inverses = bb4BatchInv(arr)

    // Verify a[i] * inv[i] = 1
    var allOne = true
    for i in 0..<n {
        let product = bb4Mul(arr[i], inverses[i])
        if product != Bb4.one {
            allOne = false
            break
        }
    }
    expect(allOne, "Bb4 batch inverse: a[i] * inv[i] = 1 for all \(n) elements")

    // Test zero handling
    var withZero = arr
    withZero[3] = .zero
    withZero[15] = .zero
    let invWithZero = bb4BatchInv(withZero)
    expect(invWithZero[3] == Bb4.zero, "Bb4 batch inv of zero = zero")
    expect(invWithZero[15] == Bb4.zero, "Bb4 batch inv of zero = zero (2)")

    // Non-zero entries still correct
    let p0 = bb4Mul(withZero[0], invWithZero[0])
    expect(p0 == Bb4.one, "Bb4 batch inv correct after zero skip")
}

// MARK: - Inner Product

private func testBb4InnerProduct() {
    let n = 8
    var seed: UInt64 = 0xABCDEF01
    let a = bb4RandomArray(count: n, seed: &seed)

    // Inner product with one-hot vector selects element
    var oneHot = [Bb4](repeating: .zero, count: n)
    oneHot[3] = Bb4.one
    let ip = bb4InnerProduct(a, oneHot)
    expect(ip == a[3], "Bb4 inner product with one-hot selects element")

    // Inner product with zeros = zero
    let zeros = [Bb4](repeating: .zero, count: n)
    let ipZero = bb4InnerProduct(a, zeros)
    expect(ipZero == Bb4.zero, "Bb4 inner product with zeros = 0")
}

// MARK: - Extension NTT

private func testBb4NTT() {
    // Test NTT round-trip: INTT(NTT(a)) = a
    let n = 8
    var seed: UInt64 = 0x42424242
    let coeffs = bb4RandomArray(count: n, seed: &seed)

    let transformed = bb4NTTForward(coeffs)
    let recovered = bb4NTTInverse(transformed)

    for i in 0..<n {
        expect(recovered[i] == coeffs[i], "Bb4 NTT round-trip[\(i)]")
    }

    // NTT of all zeros = all zeros
    let zeros = [Bb4](repeating: .zero, count: n)
    let nttZeros = bb4NTTForward(zeros)
    for i in 0..<n {
        expect(nttZeros[i] == Bb4.zero, "Bb4 NTT of zeros[\(i)]")
    }

    // NTT of [1, 0, ..., 0] = [1, 1, ..., 1]
    var delta = [Bb4](repeating: .zero, count: n)
    delta[0] = Bb4.one
    let nttDelta = bb4NTTForward(delta)
    for i in 0..<n {
        expect(nttDelta[i] == Bb4.one, "Bb4 NTT of delta[\(i)]")
    }

    // NTT of constant = [n*c, 0, ..., 0] after INTT (Parseval-like check)
    let c = Bb4(c0: Bb(v: 7), c1: Bb(v: 0), c2: Bb(v: 0), c3: Bb(v: 0))
    let constant = [Bb4](repeating: c, count: n)
    let nttConst = bb4NTTForward(constant)
    // First element should be n * c
    let nc = bb4ScalarMul(Bb(v: UInt32(n)), c)
    expect(nttConst[0] == nc, "Bb4 NTT of constant: first elem = n*c")

    // Larger NTT round-trip
    let n2 = 64
    var seed2: UInt64 = 0x98765432
    let coeffs2 = bb4RandomArray(count: n2, seed: &seed2)
    let recovered2 = bb4NTTInverse(bb4NTTForward(coeffs2))
    var allMatch = true
    for i in 0..<n2 {
        if recovered2[i] != coeffs2[i] { allMatch = false; break }
    }
    expect(allMatch, "Bb4 NTT round-trip n=64")
}

// MARK: - Polynomial Multiply via NTT

private func testBb4PolyMul() {
    // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2  (in Bb4 coefficients as Bb4 scalars)
    let a = [Bb4(from: Bb(v: 1)), Bb4(from: Bb(v: 2))]
    let b = [Bb4(from: Bb(v: 3)), Bb4(from: Bb(v: 4))]
    let c = bb4PolyMul(a, b)

    expect(c.count == 3, "Bb4 poly mul degree")
    expect(c[0] == Bb4(from: Bb(v: 3)), "Bb4 poly mul c0 = 3")
    expect(c[1] == Bb4(from: Bb(v: 10)), "Bb4 poly mul c1 = 10")
    expect(c[2] == Bb4(from: Bb(v: 8)), "Bb4 poly mul c2 = 8")

    // Multiply by zero polynomial
    let z = [Bb4.zero]
    let cz = bb4PolyMul(a, z)
    expect(cz.count == 2, "Bb4 poly mul by zero degree")
    expect(cz[0] == Bb4.zero && cz[1] == Bb4.zero, "Bb4 poly mul by zero = zero")

    // Multiply by one polynomial
    let one = [Bb4.one]
    let co = bb4PolyMul(a, one)
    expect(co.count == 2, "Bb4 poly mul by one degree")
    expect(co[0] == a[0] && co[1] == a[1], "Bb4 poly mul by one = self")

    // Cross-check: naive multiply vs NTT multiply for random polys
    let n = 8
    var seed: UInt64 = 0xAAAABBBB
    let pa = bb4RandomArray(count: n, seed: &seed)
    let pb = bb4RandomArray(count: n, seed: &seed)
    let nttResult = bb4PolyMul(pa, pb)
    let naiveResult = naiveBb4PolyMul(pa, pb)
    expect(nttResult.count == naiveResult.count, "Bb4 NTT vs naive poly mul length")
    var polyMatch = true
    for i in 0..<nttResult.count {
        if nttResult[i] != naiveResult[i] { polyMatch = false; break }
    }
    expect(polyMatch, "Bb4 NTT poly mul = naive poly mul")
}

/// Naive O(n^2) polynomial multiplication for cross-checking.
private func naiveBb4PolyMul(_ a: [Bb4], _ b: [Bb4]) -> [Bb4] {
    let n = a.count + b.count - 1
    var c = [Bb4](repeating: .zero, count: n)
    for i in 0..<a.count {
        for j in 0..<b.count {
            c[i + j] = bb4Add(c[i + j], bb4Mul(a[i], b[j]))
        }
    }
    return c
}

// MARK: - GPU Engine

private func testBb4Engine() {
    let engine = GPUBabyBearExtensionEngine.shared

    let a = Bb4(c0: Bb(v: 42), c1: Bb(v: 17), c2: Bb(v: 99), c3: Bb(v: 7))
    let b = Bb4(c0: Bb(v: 13), c1: Bb(v: 55), c2: Bb(v: 3), c3: Bb(v: 200))

    // Single ops
    expect(engine.mul(a, b) == bb4Mul(a, b), "Engine mul")
    expect(engine.sqr(a) == bb4Sqr(a), "Engine sqr")
    expect(engine.inv(a) == bb4Inverse(a), "Engine inv")
    expect(engine.frobenius(a) == bb4Frobenius(a), "Engine frobenius")

    // Batch ops
    let xs = [a, b]
    let ys = [b, a]
    let batchMul = engine.batchMul(xs, ys)
    expect(batchMul[0] == bb4Mul(a, b), "Engine batch mul[0]")
    expect(batchMul[1] == bb4Mul(b, a), "Engine batch mul[1]")

    let batchInv = engine.batchInv(xs)
    expect(bb4Mul(xs[0], batchInv[0]) == Bb4.one, "Engine batch inv[0]")
    expect(bb4Mul(xs[1], batchInv[1]) == Bb4.one, "Engine batch inv[1]")

    // Product
    let prod = engine.product([a, bb4Inverse(a)])
    expect(prod == Bb4.one, "Engine product a * inv(a) = 1")

    // NTT round-trip
    let coeffs = engine.randomElements(count: 8, seed: 0x42)
    let roundTrip = engine.nttInverse(engine.nttForward(coeffs))
    var nttOk = true
    for i in 0..<8 {
        if roundTrip[i] != coeffs[i] { nttOk = false; break }
    }
    expect(nttOk, "Engine NTT round-trip")

    // Conversion
    let bases = [Bb(v: 1), Bb(v: 2), Bb(v: 3)]
    let ext = engine.fromBase(bases)
    let back = engine.toBase(ext)
    expect(back[0] == bases[0] && back[1] == bases[1] && back[2] == bases[2],
           "Engine fromBase/toBase round-trip")
}

// MARK: - Benchmark

private func benchmarkBb4Mul() {
    let a = Bb4(c0: Bb(v: 0x12345), c1: Bb(v: 0x6789A), c2: Bb(v: 0xBCDEF), c3: Bb(v: 0x13579))
    let b = Bb4(c0: Bb(v: 0x2468A), c1: Bb(v: 0xBDF01), c2: Bb(v: 0x3579B), c3: Bb(v: 0xACE02))
    let count = 10_000
    let t0 = CFAbsoluteTimeGetCurrent()
    var acc = a
    for _ in 0..<count {
        acc = bb4Mul(acc, b)
    }
    let elapsed = CFAbsoluteTimeGetCurrent() - t0
    // Prevent dead-code elimination
    expect(!acc.isZero || acc.isZero, "Bb4 bench alive")
    let nsPerOp = elapsed * 1e9 / Double(count)
    print(String(format: "  Bb4 mul: %.0f ns/op (%.1f M/s)", nsPerOp, Double(count) / elapsed / 1e6))
}
