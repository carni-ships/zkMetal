import zkMetal
import Foundation

public func runGPUFieldExtensionTests() {
    suite("Fp2 — Quadratic Extension Arithmetic")
    testFp2BasicArithmetic()
    testFp2MulIdentity()
    testFp2Inverse()
    testFp2Conjugate()
    testFp2Norm()
    testFp2Frobenius()

    suite("Fp6 — Cubic Extension Arithmetic")
    testFp6BasicArithmetic()
    testFp6MulIdentity()
    testFp6Inverse()
    testFp6Squaring()

    suite("Fp12 — Degree-12 Extension Arithmetic")
    testFp12BasicArithmetic()
    testFp12MulIdentity()
    testFp12Inverse()
    testFp12Conjugate()
    testFp12Squaring()

    suite("Batch Extension Field Operations")
    testBatchFp2Mul()
    testBatchFp2Inv()
    testBatchFp12Mul()

    suite("GPU Field Extension Engine")
    testGPUFieldExtensionEngine()

    suite("Fp2 Performance (10K muls)")
    benchmarkFp2Mul()
}

// MARK: - Fp2 Tests

private func testFp2BasicArithmetic() {
    let a = Fp2(c0: fpFromInt(3), c1: fpFromInt(4))
    let b = Fp2(c0: fpFromInt(7), c1: fpFromInt(2))

    // Addition
    let sum = fp2Add(a, b)
    expect(fpToInt(sum.c0)[0] == 10, "Fp2 add real: 3+7=10")
    expect(fpToInt(sum.c1)[0] == 6, "Fp2 add imag: 4+2=6")

    // Subtraction
    let diff = fp2Sub(a, b)
    // 3-7 mod p is p-4, but let's check via round-trip
    let restored = fp2Add(diff, b)
    expect(restored == a, "Fp2 sub then add restores original")

    // Negation
    let neg = fp2Neg(a)
    let shouldBeZero = fp2Add(a, neg)
    expect(shouldBeZero == Fp2.zero, "Fp2 a + (-a) = 0")

    // Doubling
    let dbl = fp2Double(a)
    expect(dbl == fp2Add(a, a), "Fp2 double = add self")

    // Commutativity
    expect(fp2Add(a, b) == fp2Add(b, a), "Fp2 add commutative")
    expect(fp2Mul(a, b) == fp2Mul(b, a), "Fp2 mul commutative")
}

private func testFp2MulIdentity() {
    let a = Fp2(c0: fpFromInt(42), c1: fpFromInt(17))

    // Multiply by one
    expect(fp2Mul(a, Fp2.one) == a, "Fp2 a * 1 = a")

    // Multiply by zero
    expect(fp2Mul(a, Fp2.zero) == Fp2.zero, "Fp2 a * 0 = 0")

    // Associativity: (a*b)*c = a*(b*c)
    let b = Fp2(c0: fpFromInt(5), c1: fpFromInt(3))
    let c = Fp2(c0: fpFromInt(11), c1: fpFromInt(7))
    expect(fp2Mul(fp2Mul(a, b), c) == fp2Mul(a, fp2Mul(b, c)), "Fp2 mul associative")

    // Distributivity: a*(b+c) = a*b + a*c
    expect(fp2Mul(a, fp2Add(b, c)) == fp2Add(fp2Mul(a, b), fp2Mul(a, c)),
           "Fp2 mul distributive")
}

private func testFp2Inverse() {
    let a = Fp2(c0: fpFromInt(42), c1: fpFromInt(17))
    let aInv = fp2Inv(a)
    let product = fp2Mul(a, aInv)
    expect(product == Fp2.one, "Fp2 a * a^-1 = 1")

    // Pure imaginary
    let pu = Fp2(c0: Fp.zero, c1: fpFromInt(5))
    let puInv = fp2Inv(pu)
    expect(fp2Mul(pu, puInv) == Fp2.one, "Fp2 pure imaginary inverse")
}

private func testFp2Conjugate() {
    let a = Fp2(c0: fpFromInt(42), c1: fpFromInt(17))
    let conj = fp2Conj(a)
    expect(fpToInt(conj.c0)[0] == 42, "Fp2 conj preserves real")
    // conj.c1 should be -17 mod p
    let sum = fpAdd(conj.c1, fpFromInt(17))
    expect(fpToInt(sum)[0] == 0, "Fp2 conj negates imag")

    // a * conj(a) should be in Fp (imaginary part zero)
    let prod = fp2Mul(a, conj)
    expect(prod.c1.isZero, "Fp2 a * conj(a) is real")
}

private func testFp2Norm() {
    let a = Fp2(c0: fpFromInt(3), c1: fpFromInt(4))
    let norm = fp2Norm(a)
    // |a|^2 = 3^2 + 4^2 = 25
    expect(fpToInt(norm)[0] == 25, "Fp2 norm(3+4u) = 25")
}

private func testFp2Frobenius() {
    let a = Fp2(c0: fpFromInt(42), c1: fpFromInt(17))
    // Frobenius = conjugation for Fp2
    expect(fp2Frobenius(a) == fp2Conj(a), "Fp2 Frobenius = conjugation")
}

// MARK: - Fp6 Tests

private func testFp6BasicArithmetic() {
    let a = Fp6(c0: Fp2(c0: fpFromInt(1), c1: fpFromInt(2)),
                c1: Fp2(c0: fpFromInt(3), c1: fpFromInt(4)),
                c2: Fp2(c0: fpFromInt(5), c1: fpFromInt(6)))
    let b = Fp6(c0: Fp2(c0: fpFromInt(7), c1: fpFromInt(8)),
                c1: Fp2(c0: fpFromInt(9), c1: fpFromInt(10)),
                c2: Fp2(c0: fpFromInt(11), c1: fpFromInt(12)))

    // a + b - b = a
    expect(fp6Sub(fp6Add(a, b), b) == a, "Fp6 add/sub round-trip")

    // Commutativity
    expect(fp6Add(a, b) == fp6Add(b, a), "Fp6 add commutative")
    expect(fp6Mul(a, b) == fp6Mul(b, a), "Fp6 mul commutative")

    // Negation
    expect(fp6Add(a, fp6Neg(a)) == Fp6.zero, "Fp6 a + (-a) = 0")
}

private func testFp6MulIdentity() {
    let a = Fp6(c0: Fp2(c0: fpFromInt(42), c1: fpFromInt(17)),
                c1: Fp2(c0: fpFromInt(3), c1: fpFromInt(8)),
                c2: Fp2(c0: fpFromInt(11), c1: fpFromInt(5)))

    expect(fp6Mul(a, Fp6.one) == a, "Fp6 a * 1 = a")
    expect(fp6Mul(a, Fp6.zero) == Fp6.zero, "Fp6 a * 0 = 0")

    // Associativity
    let b = Fp6(c0: Fp2(c0: fpFromInt(2), c1: fpFromInt(1)),
                c1: Fp2(c0: fpFromInt(4), c1: fpFromInt(3)),
                c2: Fp2(c0: fpFromInt(6), c1: fpFromInt(5)))
    let c = Fp6(c0: Fp2(c0: fpFromInt(7), c1: fpFromInt(9)),
                c1: Fp2(c0: fpFromInt(1), c1: fpFromInt(2)),
                c2: Fp2(c0: fpFromInt(3), c1: fpFromInt(4)))
    expect(fp6Mul(fp6Mul(a, b), c) == fp6Mul(a, fp6Mul(b, c)), "Fp6 mul associative")
}

private func testFp6Inverse() {
    let a = Fp6(c0: Fp2(c0: fpFromInt(42), c1: fpFromInt(17)),
                c1: Fp2(c0: fpFromInt(3), c1: fpFromInt(8)),
                c2: Fp2(c0: fpFromInt(11), c1: fpFromInt(5)))
    let aInv = fp6Inv(a)
    let product = fp6Mul(a, aInv)
    expect(product == Fp6.one, "Fp6 a * a^-1 = 1")
}

private func testFp6Squaring() {
    let a = Fp6(c0: Fp2(c0: fpFromInt(7), c1: fpFromInt(13)),
                c1: Fp2(c0: fpFromInt(2), c1: fpFromInt(9)),
                c2: Fp2(c0: fpFromInt(4), c1: fpFromInt(1)))
    expect(fp6Sqr(a) == fp6Mul(a, a), "Fp6 sqr = mul self")
}

// MARK: - Fp12 Tests

private func testFp12BasicArithmetic() {
    let a0 = Fp6(c0: Fp2(c0: fpFromInt(1), c1: fpFromInt(2)),
                 c1: Fp2(c0: fpFromInt(3), c1: fpFromInt(4)),
                 c2: Fp2(c0: fpFromInt(5), c1: fpFromInt(6)))
    let a1 = Fp6(c0: Fp2(c0: fpFromInt(7), c1: fpFromInt(8)),
                 c1: Fp2(c0: fpFromInt(9), c1: fpFromInt(10)),
                 c2: Fp2(c0: fpFromInt(11), c1: fpFromInt(12)))
    let a = Fp12(c0: a0, c1: a1)

    let b0 = Fp6(c0: Fp2(c0: fpFromInt(13), c1: fpFromInt(14)),
                 c1: Fp2(c0: fpFromInt(15), c1: fpFromInt(16)),
                 c2: Fp2(c0: fpFromInt(17), c1: fpFromInt(18)))
    let b1 = Fp6(c0: Fp2(c0: fpFromInt(19), c1: fpFromInt(20)),
                 c1: Fp2(c0: fpFromInt(21), c1: fpFromInt(22)),
                 c2: Fp2(c0: fpFromInt(23), c1: fpFromInt(24)))
    let b = Fp12(c0: b0, c1: b1)

    // a + b - b = a
    expect(fp12Sub(fp12Add(a, b), b) == a, "Fp12 add/sub round-trip")

    // Commutativity
    expect(fp12Mul(a, b) == fp12Mul(b, a), "Fp12 mul commutative")

    // Negation
    expect(fp12Add(a, fp12Neg(a)) == Fp12.zero, "Fp12 a + (-a) = 0")
}

private func testFp12MulIdentity() {
    let a = Fp12(c0: Fp6(c0: Fp2(c0: fpFromInt(42), c1: fpFromInt(17)),
                         c1: Fp2(c0: fpFromInt(3), c1: fpFromInt(8)),
                         c2: Fp2(c0: fpFromInt(11), c1: fpFromInt(5))),
                 c1: Fp6(c0: Fp2(c0: fpFromInt(7), c1: fpFromInt(13)),
                         c1: Fp2(c0: fpFromInt(2), c1: fpFromInt(9)),
                         c2: Fp2(c0: fpFromInt(4), c1: fpFromInt(1))))

    expect(fp12Mul(a, Fp12.one) == a, "Fp12 a * 1 = a")
    expect(fp12Mul(a, Fp12.zero) == Fp12.zero, "Fp12 a * 0 = 0")
}

private func testFp12Inverse() {
    let a = Fp12(c0: Fp6(c0: Fp2(c0: fpFromInt(42), c1: fpFromInt(17)),
                         c1: Fp2(c0: fpFromInt(3), c1: fpFromInt(8)),
                         c2: Fp2(c0: fpFromInt(11), c1: fpFromInt(5))),
                 c1: Fp6(c0: Fp2(c0: fpFromInt(7), c1: fpFromInt(13)),
                         c1: Fp2(c0: fpFromInt(2), c1: fpFromInt(9)),
                         c2: Fp2(c0: fpFromInt(4), c1: fpFromInt(1))))
    let aInv = fp12Inv(a)
    let product = fp12Mul(a, aInv)
    expect(product == Fp12.one, "Fp12 a * a^-1 = 1")
}

private func testFp12Conjugate() {
    let a = Fp12(c0: Fp6(c0: Fp2(c0: fpFromInt(42), c1: fpFromInt(17)),
                         c1: Fp2(c0: fpFromInt(3), c1: fpFromInt(8)),
                         c2: Fp2(c0: fpFromInt(11), c1: fpFromInt(5))),
                 c1: Fp6(c0: Fp2(c0: fpFromInt(7), c1: fpFromInt(13)),
                         c1: Fp2(c0: fpFromInt(2), c1: fpFromInt(9)),
                         c2: Fp2(c0: fpFromInt(4), c1: fpFromInt(1))))
    let conj = fp12Conj(a)
    // a * conj(a) should have c1 = 0 if a is unitary, but in general
    // conj just negates c1
    expect(conj.c0 == a.c0, "Fp12 conj preserves c0")
    expect(conj.c1 == fp6Neg(a.c1), "Fp12 conj negates c1")
}

private func testFp12Squaring() {
    let a = Fp12(c0: Fp6(c0: Fp2(c0: fpFromInt(7), c1: fpFromInt(13)),
                         c1: Fp2(c0: fpFromInt(2), c1: fpFromInt(9)),
                         c2: Fp2(c0: fpFromInt(4), c1: fpFromInt(1))),
                 c1: Fp6(c0: Fp2(c0: fpFromInt(3), c1: fpFromInt(5)),
                         c1: Fp2(c0: fpFromInt(8), c1: fpFromInt(6)),
                         c2: Fp2(c0: fpFromInt(10), c1: fpFromInt(11))))
    expect(fp12Sqr(a) == fp12Mul(a, a), "Fp12 sqr = mul self")
}

// MARK: - Batch Tests

private func testBatchFp2Mul() {
    let xs = (0..<8).map { i in Fp2(c0: fpFromInt(UInt64(i + 1)), c1: fpFromInt(UInt64(i + 10))) }
    let ys = (0..<8).map { i in Fp2(c0: fpFromInt(UInt64(i + 20)), c1: fpFromInt(UInt64(i + 30))) }
    let batch = fp2BatchMul(xs, ys)
    for i in 0..<8 {
        expect(batch[i] == fp2Mul(xs[i], ys[i]), "Batch Fp2 mul[\(i)]")
    }
}

private func testBatchFp2Inv() {
    let xs = (0..<8).map { i in Fp2(c0: fpFromInt(UInt64(i + 1)), c1: fpFromInt(UInt64(i + 10))) }
    let invs = fp2BatchInv(xs)
    for i in 0..<8 {
        let product = fp2Mul(xs[i], invs[i])
        expect(product == Fp2.one, "Batch Fp2 inv[\(i)]")
    }
}

private func testBatchFp12Mul() {
    let make12 = { (seed: UInt64) -> Fp12 in
        Fp12(c0: Fp6(c0: Fp2(c0: fpFromInt(seed), c1: fpFromInt(seed + 1)),
                      c1: Fp2(c0: fpFromInt(seed + 2), c1: fpFromInt(seed + 3)),
                      c2: Fp2(c0: fpFromInt(seed + 4), c1: fpFromInt(seed + 5))),
             c1: Fp6(c0: Fp2(c0: fpFromInt(seed + 6), c1: fpFromInt(seed + 7)),
                      c1: Fp2(c0: fpFromInt(seed + 8), c1: fpFromInt(seed + 9)),
                      c2: Fp2(c0: fpFromInt(seed + 10), c1: fpFromInt(seed + 11))))
    }
    let xs = (0..<4).map { make12(UInt64($0) * 20 + 1) }
    let ys = (0..<4).map { make12(UInt64($0) * 20 + 100) }
    let batch = fp12BatchMul(xs, ys)
    for i in 0..<4 {
        expect(batch[i] == fp12Mul(xs[i], ys[i]), "Batch Fp12 mul[\(i)]")
    }
}

// MARK: - Engine Tests

private func testGPUFieldExtensionEngine() {
    let engine = GPUFieldExtensionEngine.shared

    // Engine batch Fp2 mul
    let xs = [Fp2(c0: fpFromInt(3), c1: fpFromInt(4)),
              Fp2(c0: fpFromInt(5), c1: fpFromInt(6))]
    let ys = [Fp2(c0: fpFromInt(7), c1: fpFromInt(8)),
              Fp2(c0: fpFromInt(9), c1: fpFromInt(10))]
    let results = engine.batchFp2Mul(xs, ys)
    expect(results[0] == fp2Mul(xs[0], ys[0]), "Engine Fp2 mul[0]")
    expect(results[1] == fp2Mul(xs[1], ys[1]), "Engine Fp2 mul[1]")

    // Engine batch Fp2 inv
    let invs = engine.batchFp2Inv(xs)
    expect(fp2Mul(xs[0], invs[0]) == Fp2.one, "Engine Fp2 inv[0]")
    expect(fp2Mul(xs[1], invs[1]) == Fp2.one, "Engine Fp2 inv[1]")

    // Engine Fp12 product
    let a12 = Fp12(c0: Fp6(c0: Fp2(c0: fpFromInt(2), c1: fpFromInt(3)),
                           c1: Fp2(c0: fpFromInt(4), c1: fpFromInt(5)),
                           c2: Fp2(c0: fpFromInt(6), c1: fpFromInt(7))),
                   c1: Fp6(c0: Fp2(c0: fpFromInt(8), c1: fpFromInt(9)),
                           c1: Fp2(c0: fpFromInt(10), c1: fpFromInt(11)),
                           c2: Fp2(c0: fpFromInt(12), c1: fpFromInt(13))))
    let prod = engine.fp12Product([a12, fp12Inv(a12)])
    expect(prod == Fp12.one, "Engine Fp12 product a * a^-1 = 1")
}

// MARK: - Benchmark

private func benchmarkFp2Mul() {
    let a = Fp2(c0: fpFromInt(0x123456789), c1: fpFromInt(0x987654321))
    let b = Fp2(c0: fpFromInt(0xDEADBEEF), c1: fpFromInt(0xCAFEBABE))
    let count = 10_000
    let t0 = CFAbsoluteTimeGetCurrent()
    var acc = a
    for _ in 0..<count {
        acc = fp2Mul(acc, b)
    }
    let elapsed = CFAbsoluteTimeGetCurrent() - t0
    // Prevent dead-code elimination
    expect(!acc.isZero || acc.isZero, "Fp2 bench alive")
    let nsPerOp = elapsed * 1e9 / Double(count)
    print(String(format: "  Fp2 mul: %.0f ns/op (%.1f M/s)", nsPerOp, Double(count) / elapsed / 1e6))
}
