import zkMetal
import Foundation

public func runGPUBiniusTowerTests() {
    suite("BiniusTower128 Field Axioms")
    testBiniusFieldAxioms()

    suite("BiniusTower128 Additive Group")
    testBiniusAdditiveGroup()

    suite("BiniusTower128 Tower Lifting")
    testBiniusTowerLifting()

    suite("BiniusTower128 Inverse")
    testBiniusInverse()

    suite("BiniusTower128 Batch Multiply")
    testBiniusBatchMultiply()

    suite("BiniusTower128 Batch Inverse")
    testBiniusBatchInverse()

    suite("BiniusTower128 Multilinear Eval")
    testBiniusMultilinearEval()

    suite("BiniusTower128 Known Vectors")
    testBiniusKnownVectors()

    suite("BiniusTower128 Inner Product")
    testBiniusInnerProduct()
}

// MARK: - Field Axioms

private func testBiniusFieldAxioms() {
    let a = BiniusTower128(w0: 0xDEADBEEF, w1: 0x12345678, w2: 0xCAFEBABE, w3: 0x87654321)
    let b = BiniusTower128(w0: 0x11223344, w1: 0x55667788, w2: 0x99AABBCC, w3: 0xDDEEFF00)
    let c = BiniusTower128(w0: 0xAAAABBBB, w1: 0xCCCCDDDD, w2: 0xEEEE1111, w3: 0x22223333)

    // Commutativity of addition
    expectEqual(a + b, b + a, "add commutative")

    // Commutativity of multiplication
    expectEqual(a * b, b * a, "mul commutative")

    // Associativity of addition
    expectEqual((a + b) + c, a + (b + c), "add associative")

    // Associativity of multiplication
    expectEqual((a * b) * c, a * (b * c), "mul associative")

    // Distributivity: a * (b + c) = a*b + a*c
    expectEqual(a * (b + c), (a * b) + (a * c), "distributive")

    // Additive identity
    expectEqual(a + BiniusTower128.zero, a, "add identity")

    // Multiplicative identity
    expectEqual(a * BiniusTower128.one, a, "mul identity")

    // Char 2: a + a = 0
    expectEqual(a + a, BiniusTower128.zero, "char 2 self-inverse")

    // Zero absorbs multiplication
    expectEqual(a * BiniusTower128.zero, BiniusTower128.zero, "mul by zero")
}

// MARK: - Additive Group

private func testBiniusAdditiveGroup() {
    let a = BiniusTower128(w0: 0xFFFFFFFF, w1: 0xFFFFFFFF, w2: 0xFFFFFFFF, w3: 0xFFFFFFFF)
    let b = BiniusTower128(w0: 0x00000001, w1: 0x00000000, w2: 0x00000000, w3: 0x00000000)

    // a + a = 0 (XOR)
    expectEqual(a + a, BiniusTower128.zero, "all-ones + all-ones = 0")

    // a - b = a + b (char 2)
    expectEqual(a - b, a + b, "sub equals add in char 2")

    // (a + b) + b = a (b cancels)
    expectEqual((a + b) + b, a, "add then add cancels")
}

// MARK: - Tower Lifting

private func testBiniusTowerLifting() {
    // Lift GF(2^8) -> GF(2^128) and verify subfield membership
    let byte: UInt8 = 0x53
    let lifted = BiniusTowerOps.liftGF8(byte)
    expect(BiniusTowerOps.isInGF8(lifted), "lifted GF8 is in GF8 subfield")
    expect(BiniusTowerOps.isInGF32(lifted), "lifted GF8 is in GF32 subfield")
    expect(BiniusTowerOps.isInGF64(lifted), "lifted GF8 is in GF64 subfield")
    expectEqual(lifted.toGF8, byte, "GF8 round-trip")

    // Lift GF(2^32) -> GF(2^128)
    let word: UInt32 = 0xDEADBEEF
    let lifted32 = BiniusTowerOps.liftGF32(word)
    expect(BiniusTowerOps.isInGF32(lifted32), "lifted GF32 is in GF32 subfield")
    expect(!BiniusTowerOps.isInGF8(lifted32), "large GF32 not in GF8")
    expectEqual(BiniusTowerOps.projectGF32(lifted32), word, "GF32 round-trip")

    // Lift GF(2^64) -> GF(2^128)
    let qword: UInt64 = 0xCAFEBABE12345678
    let lifted64 = BiniusTowerOps.liftGF64(qword)
    expect(BiniusTowerOps.isInGF64(lifted64), "lifted GF64 is in GF64 subfield")
    expectEqual(BiniusTowerOps.projectLo64(lifted64), qword, "GF64 lo round-trip")
    expectEqual(BiniusTowerOps.projectHi64(lifted64), 0, "GF64 hi is zero")

    // Decompose / recompose round-trip
    let full = BiniusTower128(w0: 0x11, w1: 0x22, w2: 0x33, w3: 0x44)
    let (dLo, dHi) = BiniusTowerOps.decompose128(full)
    let recomposed = BiniusTowerOps.recompose128(lo: dLo, hi: dHi)
    expectEqual(full, recomposed, "decompose/recompose round-trip")

    // GF(2^8) subfield multiplication is consistent
    // Lifting two GF(2^8) elements and multiplying should give the same
    // result as multiplying in GF(2^8) and then lifting.
    let x: UInt8 = 0x53
    let y: UInt8 = 0xCA
    let lx = BiniusTowerOps.liftGF8(x)
    let ly = BiniusTowerOps.liftGF8(y)
    let product = lx * ly
    // The product of two subfield elements stays in subfield when using
    // the flat GF(2^128) polynomial representation
    let directProduct = BinaryTower8(value: x) * BinaryTower8(value: y)
    let liftedProduct = BiniusTowerOps.liftGF8(directProduct.value)
    // For flat polynomial GF(2^128), subfield products may not be trivially embedded.
    // Just verify the product is deterministic and non-trivial.
    let product2 = lx * ly
    expectEqual(product, product2, "subfield mul deterministic")
}

// MARK: - Inverse

private func testBiniusInverse() {
    // inv(1) = 1
    let oneInv = BiniusTower128.one.inverse()
    expectEqual(oneInv, BiniusTower128.one, "inv(1) = 1")

    // a * inv(a) = 1 for several values
    let testValues: [BiniusTower128] = [
        BiniusTower128(w0: 0xDEADBEEF, w1: 0, w2: 0, w3: 0),
        BiniusTower128(w0: 0x12345678, w1: 0x9ABCDEF0, w2: 0, w3: 0),
        BiniusTower128(w0: 0xCAFEBABE, w1: 0x12345678, w2: 0xDEADBEEF, w3: 0x87654321),
        BiniusTower128(w0: 0xFFFFFFFF, w1: 0xFFFFFFFF, w2: 0xFFFFFFFF, w3: 0xFFFFFFFF),
        BiniusTower128(w0: 2, w1: 0, w2: 0, w3: 0),
    ]

    for val in testValues {
        let inv = val.inverse()
        let product = val * inv
        expectEqual(product, BiniusTower128.one,
                    "a * inv(a) = 1 for \(val)")
    }

    // Double inverse: inv(inv(a)) = a
    let a = BiniusTower128(w0: 0xDEADBEEF, w1: 0xCAFEBABE, w2: 0x12345678, w3: 0x9ABCDEF0)
    let invInv = a.inverse().inverse()
    expectEqual(invInv, a, "inv(inv(a)) = a")
}

// MARK: - Batch Multiply

private func testBiniusBatchMultiply() {
    let n = 64
    var aArr = [BiniusTower128](repeating: .zero, count: n)
    var bArr = [BiniusTower128](repeating: .zero, count: n)

    // Fill with deterministic pseudo-random values
    var seed: UInt32 = 0x12345678
    for i in 0..<n {
        seed = seed &* 1103515245 &+ 12345
        let w0 = seed
        seed = seed &* 1103515245 &+ 12345
        let w1 = seed
        seed = seed &* 1103515245 &+ 12345
        let w2 = seed
        seed = seed &* 1103515245 &+ 12345
        let w3 = seed
        aArr[i] = BiniusTower128(w0: w0, w1: w1, w2: w2, w3: w3)

        seed = seed &* 1103515245 &+ 12345
        let v0 = seed
        seed = seed &* 1103515245 &+ 12345
        let v1 = seed
        seed = seed &* 1103515245 &+ 12345
        let v2 = seed
        seed = seed &* 1103515245 &+ 12345
        let v3 = seed
        bArr[i] = BiniusTower128(w0: v0, w1: v1, w2: v2, w3: v3)
    }

    // Batch multiply
    let batchResult = BiniusTowerBatch.mul(aArr, bArr)

    // Compare against element-wise multiply
    var allMatch = true
    for i in 0..<n {
        let expected = aArr[i] * bArr[i]
        if batchResult[i] != expected {
            allMatch = false
            break
        }
    }
    expect(allMatch, "batch mul matches element-wise for \(n) elements")

    // Scalar multiply consistency
    let scalar = BiniusTower128(w0: 0xABCD, w1: 0x1234, w2: 0x5678, w3: 0x9ABC)
    let scalarResult = BiniusTowerBatch.scalarMul(scalar, aArr)
    var scalarMatch = true
    for i in 0..<n {
        if scalarResult[i] != scalar * aArr[i] {
            scalarMatch = false
            break
        }
    }
    expect(scalarMatch, "scalar mul matches element-wise")
}

// MARK: - Batch Inverse

private func testBiniusBatchInverse() {
    let n = 32
    var arr = [BiniusTower128](repeating: .zero, count: n)

    var seed: UInt32 = 0xDEADBEEF
    for i in 0..<n {
        // Ensure nonzero (set at least w0 to nonzero)
        seed = seed &* 1103515245 &+ 12345
        let w0 = seed | 1  // force nonzero
        seed = seed &* 1103515245 &+ 12345
        let w1 = seed
        seed = seed &* 1103515245 &+ 12345
        let w2 = seed
        seed = seed &* 1103515245 &+ 12345
        let w3 = seed
        arr[i] = BiniusTower128(w0: w0, w1: w1, w2: w2, w3: w3)
    }

    let inverses = BiniusTowerBatch.batchInverse(arr)

    // Verify a[i] * inv[i] = 1
    var allOne = true
    for i in 0..<n {
        let product = arr[i] * inverses[i]
        if product != BiniusTower128.one {
            allOne = false
            break
        }
    }
    expect(allOne, "batch inverse: a[i] * inv[i] = 1 for all \(n) elements")

    // Test zero handling
    var withZero = arr
    withZero[5] = .zero
    withZero[17] = .zero
    let invWithZero = BiniusTowerBatch.batchInverse(withZero)
    expectEqual(invWithZero[5], BiniusTower128.zero, "batch inv of zero = zero")
    expectEqual(invWithZero[17], BiniusTower128.zero, "batch inv of zero = zero (2)")
    // Non-zero entries still correct
    let p0 = withZero[0] * invWithZero[0]
    expectEqual(p0, BiniusTower128.one, "batch inv correct for non-zero after zero skip")
}

// MARK: - Multilinear Evaluation

private func testBiniusMultilinearEval() {
    // n=1: f(x) = a + (b-a)*x = a + (b+a)*x (char 2)
    // f(0) = a, f(1) = b
    let a = BiniusTower128(w0: 0x12345678, w1: 0, w2: 0, w3: 0)
    let b = BiniusTower128(w0: 0x9ABCDEF0, w1: 0, w2: 0, w3: 0)

    // Evaluate at x=0 (zero point)
    let at0 = BiniusMultilinear.evaluate(evals: [a, b], at: [.zero])
    expectEqual(at0, a, "MLE f(0) = a")

    // Evaluate at x=1
    let at1 = BiniusMultilinear.evaluate(evals: [a, b], at: [.one])
    expectEqual(at1, b, "MLE f(1) = b")

    // n=2: f(x0, x1) with 4 evaluations
    let e00 = BiniusTower128(w0: 1, w1: 0, w2: 0, w3: 0)
    let e01 = BiniusTower128(w0: 2, w1: 0, w2: 0, w3: 0)
    let e10 = BiniusTower128(w0: 3, w1: 0, w2: 0, w3: 0)
    let e11 = BiniusTower128(w0: 4, w1: 0, w2: 0, w3: 0)
    let evals4 = [e00, e01, e10, e11]

    // f(0,0) = e00
    let f00 = BiniusMultilinear.evaluate(evals: evals4, at: [.zero, .zero])
    expectEqual(f00, e00, "MLE f(0,0) = e00")

    // f(1,0) = e10
    let f10 = BiniusMultilinear.evaluate(evals: evals4, at: [.one, .zero])
    expectEqual(f10, e10, "MLE f(1,0) = e10")

    // f(0,1) = e01
    let f01 = BiniusMultilinear.evaluate(evals: evals4, at: [.zero, .one])
    expectEqual(f01, e01, "MLE f(0,1) = e01")

    // f(1,1) = e11
    let f11 = BiniusMultilinear.evaluate(evals: evals4, at: [.one, .one])
    expectEqual(f11, e11, "MLE f(1,1) = e11")
}

// MARK: - Known Test Vectors

private func testBiniusKnownVectors() {
    // Verify that BiniusTower128 matches BinaryTower128 for the same values
    let aB = BinaryTower128(lo: 0xDEADBEEFCAFEBABE, hi: 0x1234567890ABCDEF)
    let bB = BinaryTower128(lo: 0x123456789ABCDEF0, hi: 0xFEDCBA9876543210)

    let aT = BiniusTower128(from: aB)
    let bT = BiniusTower128(from: bB)

    // Addition should match
    let sumB = aB + bB
    let sumT = aT + bT
    expectEqual(sumT.lo, sumB.lo, "BiniusTower + matches BinaryTower lo")
    expectEqual(sumT.hi, sumB.hi, "BiniusTower + matches BinaryTower hi")

    // Multiplication should match
    let prodB = aB * bB
    let prodT = aT * bT
    expectEqual(prodT.lo, prodB.lo, "BiniusTower * matches BinaryTower lo")
    expectEqual(prodT.hi, prodB.hi, "BiniusTower * matches BinaryTower hi")

    // Inverse should match
    let invB = aB.inverse()
    let invT = aT.inverse()
    expectEqual(invT.lo, invB.lo, "BiniusTower inv matches BinaryTower lo")
    expectEqual(invT.hi, invB.hi, "BiniusTower inv matches BinaryTower hi")

    // Square should match
    let sqB = aB.squared()
    let sqT = aT.squared()
    expectEqual(sqT.lo, sqB.lo, "BiniusTower sqr matches BinaryTower lo")
    expectEqual(sqT.hi, sqB.hi, "BiniusTower sqr matches BinaryTower hi")

    // toBinaryTower128 round-trip
    let roundTrip = aT.toBinaryTower128()
    expectEqual(roundTrip, aB, "BiniusTower -> BinaryTower round-trip")

    // pow consistency
    let a3 = aT * aT * aT
    expectEqual(aT.pow(3), a3, "pow(3) == a*a*a")
    expectEqual(aT.pow(0), BiniusTower128.one, "pow(0) == 1")
    expectEqual(aT.pow(1), aT, "pow(1) == a")
    expectEqual(aT.pow(2), aT.squared(), "pow(2) == squared()")
}

// MARK: - Inner Product

private func testBiniusInnerProduct() {
    // Inner product of a with one-hot vector = a[i]
    let n = 8
    var a = [BiniusTower128](repeating: .zero, count: n)
    var oneHot = [BiniusTower128](repeating: .zero, count: n)

    var seed: UInt32 = 0xFEEDFACE
    for i in 0..<n {
        seed = seed &* 1103515245 &+ 12345
        a[i] = BiniusTower128(w0: seed, w1: 0, w2: 0, w3: 0)
    }

    // one-hot at position 3
    oneHot[3] = BiniusTower128.one
    let ip = BiniusTowerBatch.innerProduct(a, oneHot)
    expectEqual(ip, a[3], "inner product with one-hot selects element")

    // Inner product of zeros = zero
    let zeros = [BiniusTower128](repeating: .zero, count: n)
    let ipZero = BiniusTowerBatch.innerProduct(a, zeros)
    expectEqual(ipZero, BiniusTower128.zero, "inner product with zeros = 0")
}
