import zkMetal
import Foundation

public func runGPUGoldilocksExtensionTests() {
    suite("Gl2 — Quadratic Extension Basic Arithmetic")
    testGl2AddSub()
    testGl2Negation()
    testGl2Doubling()
    testGl2Commutativity()

    suite("Gl2 — Multiplication")
    testGl2MulIdentity()
    testGl2MulZero()
    testGl2MulAssociativity()
    testGl2MulDistributivity()
    testGl2MulKnownValues()

    suite("Gl2 — Squaring")
    testGl2SqrMatchesMul()

    suite("Gl2 — Inverse")
    testGl2Inverse()
    testGl2InversePureExtension()
    testGl2InverseBaseField()

    suite("Gl2 — Exponentiation")
    testGl2Pow()

    suite("Gl2 — Frobenius & Conjugate")
    testGl2Conjugate()
    testGl2Frobenius()
    testGl2FrobeniusSquaredIsIdentity()

    suite("Gl2 — Norm")
    testGl2Norm()

    suite("Gl2 — Conversion (Gl <-> Gl2)")
    testGl2Conversion()
    testGl2IsInBaseField()

    suite("Gl2 — Batch Operations")
    testBatchGl2Add()
    testBatchGl2Sub()
    testBatchGl2Mul()
    testBatchGl2Sqr()
    testBatchGl2Inv()
    testBatchGl2ScalarMul()

    suite("Gl2 — Extension NTT")
    testGl2NTTRoundTrip()
    testGl2NTTConvolution()
    testGl2NTTSizes()

    suite("Gl2 — Random Generation")
    testGl2Random()

    suite("Gl2 — GPUGoldilocksExtensionEngine")
    testEngineArithmetic()
    testEngineBatch()
    testEngineNTT()
    testEngineReductions()
    testEngineConversion()

    suite("Gl2 — Performance (10K muls)")
    benchmarkGl2Mul()
}

// MARK: - Basic Arithmetic

private func testGl2AddSub() {
    let a = Gl2(c0: glFromInt(3), c1: glFromInt(4))
    let b = Gl2(c0: glFromInt(7), c1: glFromInt(2))

    // Addition
    let sum = gl2Add(a, b)
    expect(sum.c0.v == 10, "Gl2 add c0: 3+7=10")
    expect(sum.c1.v == 6, "Gl2 add c1: 4+2=6")

    // Subtraction round-trip
    let diff = gl2Sub(a, b)
    let restored = gl2Add(diff, b)
    expect(restored == a, "Gl2 sub then add restores original")

    // a - a = 0
    expect(gl2Sub(a, a) == Gl2.zero, "Gl2 a - a = 0")
}

private func testGl2Negation() {
    let a = Gl2(c0: glFromInt(42), c1: glFromInt(17))
    let neg = gl2Neg(a)
    let shouldBeZero = gl2Add(a, neg)
    expect(shouldBeZero == Gl2.zero, "Gl2 a + (-a) = 0")

    // -0 = 0
    expect(gl2Neg(Gl2.zero) == Gl2.zero, "Gl2 -0 = 0")
}

private func testGl2Doubling() {
    let a = Gl2(c0: glFromInt(42), c1: glFromInt(17))
    let dbl = gl2Double(a)
    expect(dbl == gl2Add(a, a), "Gl2 double = add self")
}

private func testGl2Commutativity() {
    let a = Gl2(c0: glFromInt(13), c1: glFromInt(29))
    let b = Gl2(c0: glFromInt(7), c1: glFromInt(43))

    expect(gl2Add(a, b) == gl2Add(b, a), "Gl2 add commutative")
    expect(gl2Mul(a, b) == gl2Mul(b, a), "Gl2 mul commutative")
}

// MARK: - Multiplication

private func testGl2MulIdentity() {
    let a = Gl2(c0: glFromInt(42), c1: glFromInt(17))
    expect(gl2Mul(a, Gl2.one) == a, "Gl2 a * 1 = a")
}

private func testGl2MulZero() {
    let a = Gl2(c0: glFromInt(42), c1: glFromInt(17))
    expect(gl2Mul(a, Gl2.zero) == Gl2.zero, "Gl2 a * 0 = 0")
}

private func testGl2MulAssociativity() {
    let a = Gl2(c0: glFromInt(5), c1: glFromInt(3))
    let b = Gl2(c0: glFromInt(11), c1: glFromInt(7))
    let c = Gl2(c0: glFromInt(13), c1: glFromInt(2))
    expect(gl2Mul(gl2Mul(a, b), c) == gl2Mul(a, gl2Mul(b, c)), "Gl2 mul associative")
}

private func testGl2MulDistributivity() {
    let a = Gl2(c0: glFromInt(5), c1: glFromInt(3))
    let b = Gl2(c0: glFromInt(11), c1: glFromInt(7))
    let c = Gl2(c0: glFromInt(13), c1: glFromInt(2))
    let lhs = gl2Mul(a, gl2Add(b, c))
    let rhs = gl2Add(gl2Mul(a, b), gl2Mul(a, c))
    expect(lhs == rhs, "Gl2 mul distributive: a*(b+c) = a*b + a*c")
}

private func testGl2MulKnownValues() {
    // (1 + 1*x)^2 = 1 + 2x + x^2 = 1 + 7 + 2x = 8 + 2x (since x^2 = 7)
    let a = Gl2(c0: glFromInt(1), c1: glFromInt(1))
    let sq = gl2Mul(a, a)
    expect(sq.c0.v == 8, "Gl2 (1+x)^2 c0 = 8")
    expect(sq.c1.v == 2, "Gl2 (1+x)^2 c1 = 2")

    // (0 + 1*x) * (0 + 1*x) = x^2 = 7 + 0*x
    let x = Gl2(c0: glFromInt(0), c1: glFromInt(1))
    let xsq = gl2Mul(x, x)
    expect(xsq.c0.v == 7, "Gl2 x^2 = 7")
    expect(xsq.c1.v == 0, "Gl2 x^2 has no extension part")

    // (2 + 3x)(4 + 5x) = 8 + 10x + 12x + 15*x^2 = 8 + 105 + 22x = 113 + 22x
    let p = Gl2(c0: glFromInt(2), c1: glFromInt(3))
    let q = Gl2(c0: glFromInt(4), c1: glFromInt(5))
    let pq = gl2Mul(p, q)
    expect(pq.c0.v == 113, "Gl2 (2+3x)(4+5x) c0 = 8 + 7*15 = 113")
    expect(pq.c1.v == 22, "Gl2 (2+3x)(4+5x) c1 = 10+12 = 22")
}

// MARK: - Squaring

private func testGl2SqrMatchesMul() {
    let a = Gl2(c0: glFromInt(42), c1: glFromInt(17))
    expect(gl2Sqr(a) == gl2Mul(a, a), "Gl2 sqr = mul self")

    let b = Gl2(c0: glFromInt(0), c1: glFromInt(99))
    expect(gl2Sqr(b) == gl2Mul(b, b), "Gl2 sqr pure extension = mul self")

    let c = Gl2(c0: glFromInt(123456789), c1: glFromInt(987654321))
    expect(gl2Sqr(c) == gl2Mul(c, c), "Gl2 sqr large values = mul self")
}

// MARK: - Inverse

private func testGl2Inverse() {
    let a = Gl2(c0: glFromInt(42), c1: glFromInt(17))
    let aInv = gl2Inv(a)
    let product = gl2Mul(a, aInv)
    expect(product == Gl2.one, "Gl2 a * a^-1 = 1")
}

private func testGl2InversePureExtension() {
    let a = Gl2(c0: .zero, c1: glFromInt(5))
    let aInv = gl2Inv(a)
    let product = gl2Mul(a, aInv)
    expect(product == Gl2.one, "Gl2 pure extension inverse: (5x)^-1 * 5x = 1")
}

private func testGl2InverseBaseField() {
    let a = Gl2(c0: glFromInt(42), c1: .zero)
    let aInv = gl2Inv(a)
    let product = gl2Mul(a, aInv)
    expect(product == Gl2.one, "Gl2 base field inverse")
    // Should be equivalent to base field inverse
    expect(aInv.c1.isZero, "Gl2 inverse of base element stays in base field")
}

// MARK: - Exponentiation

private func testGl2Pow() {
    let a = Gl2(c0: glFromInt(3), c1: glFromInt(5))

    // a^0 = 1
    expect(gl2Pow(a, 0) == Gl2.one, "Gl2 a^0 = 1")

    // a^1 = a
    expect(gl2Pow(a, 1) == a, "Gl2 a^1 = a")

    // a^2 = a*a
    expect(gl2Pow(a, 2) == gl2Mul(a, a), "Gl2 a^2 = a*a")

    // a^4 = (a^2)^2
    expect(gl2Pow(a, 4) == gl2Sqr(gl2Sqr(a)), "Gl2 a^4 = (a^2)^2")

    // a * a^(p^2 - 2) = 1  (Fermat's little theorem for Gl2, |Gl2*| = p^2 - 1)
    // Checked indirectly via inv
    let aInv = gl2Inv(a)
    expect(gl2Mul(a, aInv) == Gl2.one, "Gl2 a * a^-1 = 1 (Fermat)")
}

// MARK: - Frobenius & Conjugate

private func testGl2Conjugate() {
    let a = Gl2(c0: glFromInt(42), c1: glFromInt(17))
    let conj = gl2Conj(a)
    expect(conj.c0 == a.c0, "Gl2 conj preserves c0")
    expect(glAdd(conj.c1, a.c1).v == 0, "Gl2 conj negates c1")
}

private func testGl2Frobenius() {
    let a = Gl2(c0: glFromInt(42), c1: glFromInt(17))
    // Frobenius = conjugation for degree-2 extension
    expect(gl2Frobenius(a) == gl2Conj(a), "Gl2 Frobenius = conjugation")
}

private func testGl2FrobeniusSquaredIsIdentity() {
    // Frobenius^2 should be identity for Gl2/Gl (degree 2 extension)
    let a = Gl2(c0: glFromInt(42), c1: glFromInt(17))
    let frob2 = gl2Frobenius(gl2Frobenius(a))
    expect(frob2 == a, "Gl2 Frobenius^2 = identity")
}

// MARK: - Norm

private func testGl2Norm() {
    // norm(a + bx) = a^2 - 7*b^2
    let a = Gl2(c0: glFromInt(10), c1: glFromInt(3))
    let norm = gl2Norm(a)
    // 10^2 - 7*3^2 = 100 - 63 = 37
    expect(norm.v == 37, "Gl2 norm(10+3x) = 37")

    // norm(a) * norm(b) = norm(a*b) (multiplicativity)
    let b = Gl2(c0: glFromInt(5), c1: glFromInt(2))
    let normA = gl2Norm(a)
    let normB = gl2Norm(b)
    let normAB = gl2Norm(gl2Mul(a, b))
    expect(glMul(normA, normB) == normAB, "Gl2 norm multiplicative: N(a)*N(b) = N(a*b)")
}

// MARK: - Conversion

private func testGl2Conversion() {
    let g = glFromInt(42)
    let g2 = gl2FromGl(g)
    expect(g2.c0 == g, "Gl -> Gl2 preserves c0")
    expect(g2.c1.isZero, "Gl -> Gl2 sets c1 = 0")

    let back = gl2ToGl(g2)
    expect(back == g, "Gl2 -> Gl round-trip")
}

private func testGl2IsInBaseField() {
    let base = gl2FromGl(glFromInt(42))
    expect(gl2IsInBaseField(base), "Gl2 from base field is in base field")

    let ext = Gl2(c0: glFromInt(42), c1: glFromInt(17))
    expect(!gl2IsInBaseField(ext), "Gl2 with c1 != 0 is not in base field")

    expect(gl2IsInBaseField(Gl2.zero), "Gl2 zero is in base field")
    expect(gl2IsInBaseField(Gl2.one), "Gl2 one is in base field")
}

// MARK: - Batch Operations

private func testBatchGl2Add() {
    let xs = (0..<8).map { i in Gl2(c0: glFromInt(UInt64(i + 1)), c1: glFromInt(UInt64(i + 10))) }
    let ys = (0..<8).map { i in Gl2(c0: glFromInt(UInt64(i + 20)), c1: glFromInt(UInt64(i + 30))) }
    let batch = gl2BatchAdd(xs, ys)
    for i in 0..<8 {
        expect(batch[i] == gl2Add(xs[i], ys[i]), "Batch Gl2 add[\(i)]")
    }
}

private func testBatchGl2Sub() {
    let xs = (0..<8).map { i in Gl2(c0: glFromInt(UInt64(i + 20)), c1: glFromInt(UInt64(i + 30))) }
    let ys = (0..<8).map { i in Gl2(c0: glFromInt(UInt64(i + 1)), c1: glFromInt(UInt64(i + 10))) }
    let batch = gl2BatchSub(xs, ys)
    for i in 0..<8 {
        expect(batch[i] == gl2Sub(xs[i], ys[i]), "Batch Gl2 sub[\(i)]")
    }
}

private func testBatchGl2Mul() {
    let xs = (0..<8).map { i in Gl2(c0: glFromInt(UInt64(i + 1)), c1: glFromInt(UInt64(i + 10))) }
    let ys = (0..<8).map { i in Gl2(c0: glFromInt(UInt64(i + 20)), c1: glFromInt(UInt64(i + 30))) }
    let batch = gl2BatchMul(xs, ys)
    for i in 0..<8 {
        expect(batch[i] == gl2Mul(xs[i], ys[i]), "Batch Gl2 mul[\(i)]")
    }
}

private func testBatchGl2Sqr() {
    let xs = (0..<8).map { i in Gl2(c0: glFromInt(UInt64(i + 1)), c1: glFromInt(UInt64(i + 10))) }
    let batch = gl2BatchSqr(xs)
    for i in 0..<8 {
        expect(batch[i] == gl2Sqr(xs[i]), "Batch Gl2 sqr[\(i)]")
    }
}

private func testBatchGl2Inv() {
    let xs = (0..<8).map { i in Gl2(c0: glFromInt(UInt64(i + 1)), c1: glFromInt(UInt64(i + 10))) }
    let invs = gl2BatchInv(xs)
    for i in 0..<8 {
        let product = gl2Mul(xs[i], invs[i])
        expect(product == Gl2.one, "Batch Gl2 inv[\(i)]")
    }
}

private func testBatchGl2ScalarMul() {
    let scalars = (0..<8).map { i in glFromInt(UInt64(i + 2)) }
    let xs = (0..<8).map { i in Gl2(c0: glFromInt(UInt64(i + 1)), c1: glFromInt(UInt64(i + 10))) }
    let batch = gl2BatchScalarMul(scalars, xs)
    for i in 0..<8 {
        expect(batch[i] == gl2ScalarMul(scalars[i], xs[i]), "Batch Gl2 scalar mul[\(i)]")
    }
}

// MARK: - NTT

private func testGl2NTTRoundTrip() {
    // NTT then INTT should return the original
    let n = 8
    let input = (0..<n).map { i in Gl2(c0: glFromInt(UInt64(i + 1)), c1: glFromInt(UInt64(i + 100))) }
    let transformed = gl2NTT(input)
    let recovered = gl2INTT(transformed)
    for i in 0..<n {
        expect(recovered[i] == input[i], "Gl2 NTT round-trip[\(i)]")
    }
}

private func testGl2NTTConvolution() {
    // Polynomial multiplication via NTT:
    // f(x) = 1 + 2x, g(x) = 3 + 4x
    // f*g = 3 + 10x + 8x^2 (in polynomial ring)
    // Embed in size 4 to avoid wrap-around
    let n = 4
    var f = [Gl2](repeating: .zero, count: n)
    var g = [Gl2](repeating: .zero, count: n)
    f[0] = gl2FromGl(glFromInt(1))
    f[1] = gl2FromGl(glFromInt(2))
    g[0] = gl2FromGl(glFromInt(3))
    g[1] = gl2FromGl(glFromInt(4))

    let fHat = gl2NTT(f)
    let gHat = gl2NTT(g)
    // Pointwise multiply
    var hHat = [Gl2](repeating: .zero, count: n)
    for i in 0..<n {
        hHat[i] = gl2Mul(fHat[i], gHat[i])
    }
    let h = gl2INTT(hHat)

    // h should be [3, 10, 8, 0]
    expect(h[0].c0.v == 3 && h[0].c1.isZero, "Gl2 NTT convolution h[0] = 3")
    expect(h[1].c0.v == 10 && h[1].c1.isZero, "Gl2 NTT convolution h[1] = 10")
    expect(h[2].c0.v == 8 && h[2].c1.isZero, "Gl2 NTT convolution h[2] = 8")
    expect(h[3] == Gl2.zero, "Gl2 NTT convolution h[3] = 0")
}

private func testGl2NTTSizes() {
    // Test various power-of-2 sizes
    for logN in [1, 2, 3, 4, 5] {
        let n = 1 << logN
        var input = [Gl2]()
        for i in 0..<n {
            input.append(Gl2(c0: glFromInt(UInt64(i + 1)), c1: glFromInt(UInt64(i * 3 + 7))))
        }
        let recovered = gl2INTT(gl2NTT(input))
        var ok = true
        for i in 0..<n {
            if recovered[i] != input[i] { ok = false; break }
        }
        expect(ok, "Gl2 NTT round-trip size \(n)")
    }
}

// MARK: - Random Generation

private func testGl2Random() {
    let elems = gl2RandomBatch(100)
    expect(elems.count == 100, "Gl2 random batch count = 100")

    // All elements should be valid (c0, c1 < p)
    var allValid = true
    for e in elems {
        if e.c0.v >= Gl.P || e.c1.v >= Gl.P {
            allValid = false
            break
        }
    }
    expect(allValid, "Gl2 random elements all < p")

    // Not all the same (probabilistic, but 100 identical random elements is impossible)
    let allSame = elems.allSatisfy { $0 == elems[0] }
    expect(!allSame, "Gl2 random elements are not all identical")
}

// MARK: - Engine Tests

private func testEngineArithmetic() {
    let engine = GPUGoldilocksExtensionEngine.shared
    let a = Gl2(c0: glFromInt(42), c1: glFromInt(17))
    let b = Gl2(c0: glFromInt(7), c1: glFromInt(3))

    expect(engine.add(a, b) == gl2Add(a, b), "Engine add")
    expect(engine.sub(a, b) == gl2Sub(a, b), "Engine sub")
    expect(engine.mul(a, b) == gl2Mul(a, b), "Engine mul")
    expect(engine.sqr(a) == gl2Sqr(a), "Engine sqr")
    expect(engine.inv(a) == gl2Inv(a), "Engine inv")
    expect(engine.frobenius(a) == gl2Frobenius(a), "Engine frobenius")
}

private func testEngineBatch() {
    let engine = GPUGoldilocksExtensionEngine.shared
    let xs = (0..<4).map { i in Gl2(c0: glFromInt(UInt64(i + 1)), c1: glFromInt(UInt64(i + 10))) }
    let ys = (0..<4).map { i in Gl2(c0: glFromInt(UInt64(i + 20)), c1: glFromInt(UInt64(i + 30))) }

    let adds = engine.batchAdd(xs, ys)
    let muls = engine.batchMul(xs, ys)
    let invs = engine.batchInv(xs)
    for i in 0..<4 {
        expect(adds[i] == gl2Add(xs[i], ys[i]), "Engine batch add[\(i)]")
        expect(muls[i] == gl2Mul(xs[i], ys[i]), "Engine batch mul[\(i)]")
        expect(gl2Mul(xs[i], invs[i]) == Gl2.one, "Engine batch inv[\(i)]")
    }
}

private func testEngineNTT() {
    let engine = GPUGoldilocksExtensionEngine.shared
    let n = 8
    let input = (0..<n).map { i in Gl2(c0: glFromInt(UInt64(i + 1)), c1: glFromInt(UInt64(i + 50))) }
    let recovered = engine.intt(engine.ntt(input))
    for i in 0..<n {
        expect(recovered[i] == input[i], "Engine NTT round-trip[\(i)]")
    }
}

private func testEngineReductions() {
    let engine = GPUGoldilocksExtensionEngine.shared
    let a = Gl2(c0: glFromInt(3), c1: glFromInt(5))
    let b = Gl2(c0: glFromInt(7), c1: glFromInt(11))

    // Product
    let prod = engine.product([a, b, gl2Inv(b)])
    expect(prod == a, "Engine product a * b * b^-1 = a")

    // Sum
    let s = engine.sum([a, b])
    expect(s == gl2Add(a, b), "Engine sum")

    // Inner product
    let ip = engine.innerProduct([a, b], [Gl2.one, Gl2.one])
    expect(ip == gl2Add(a, b), "Engine inner product with ones = sum")
}

private func testEngineConversion() {
    let engine = GPUGoldilocksExtensionEngine.shared
    let gls = [glFromInt(1), glFromInt(2), glFromInt(3)]
    let gl2s = engine.fromBaseField(gls)
    expect(gl2s.count == 3, "Engine fromBaseField count")
    for i in 0..<3 {
        expect(gl2s[i].c0 == gls[i], "Engine fromBaseField[\(i)] c0")
        expect(gl2s[i].c1.isZero, "Engine fromBaseField[\(i)] c1 = 0")
    }
    let back = engine.toBaseField(gl2s)
    for i in 0..<3 {
        expect(back[i] == gls[i], "Engine toBaseField round-trip[\(i)]")
    }
}

// MARK: - Benchmark

private func benchmarkGl2Mul() {
    let a = Gl2(c0: Gl(v: 0x123456789), c1: Gl(v: 0x987654321))
    let b = Gl2(c0: Gl(v: 0xDEADBEEF), c1: Gl(v: 0xCAFEBABE))
    let count = 10_000
    let t0 = CFAbsoluteTimeGetCurrent()
    var acc = a
    for _ in 0..<count {
        acc = gl2Mul(acc, b)
    }
    let elapsed = CFAbsoluteTimeGetCurrent() - t0
    expect(!acc.isZero || acc.isZero, "Gl2 bench alive")
    let nsPerOp = elapsed * 1e9 / Double(count)
    print(String(format: "  Gl2 mul: %.0f ns/op (%.1f M/s)", nsPerOp, Double(count) / elapsed / 1e6))
}
