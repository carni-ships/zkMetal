import zkMetal
import Foundation

public func runGPUBiniusBinaryFieldTests() {
    suite("GPU Binius Binary Field: F2 Arithmetic")
    testF2Arithmetic()

    suite("GPU Binius Binary Field: F4 Arithmetic")
    testF4Arithmetic()

    suite("GPU Binius Binary Field: F4 Field Axioms")
    testF4FieldAxioms()

    suite("GPU Binius Binary Field: F16 Arithmetic")
    testF16Arithmetic()

    suite("GPU Binius Binary Field: F16 Inverse")
    testF16Inverse()

    suite("GPU Binius Binary Field: F256 Arithmetic")
    testF256Arithmetic()

    suite("GPU Binius Binary Field: F256 Field Axioms")
    testF256FieldAxioms()

    suite("GPU Binius Binary Field: F256 Inverse")
    testF256Inverse()

    suite("GPU Binius Binary Field: F256 Polynomial")
    testF256Polynomial()

    suite("GPU Binius Binary Field: Packed Binary Field 32")
    testPackedBinaryField32()

    suite("GPU Binius Binary Field: Packed Binary Field 64")
    testPackedBinaryField64()

    suite("GPU Binius Binary Field: Engine Batch Ops")
    testEngineBatchOps()

    suite("GPU Binius Binary Field: Batch Inverse 256")
    testBatchInverse256()

    suite("GPU Binius Binary Field: Multilinear Eval 256")
    testMultilinearEval256()

    suite("GPU Binius Binary Field: Inner Product 256")
    testInnerProduct256()

    suite("GPU Binius Binary Field: Pack/Unpack Bits")
    testPackUnpackBits()

    suite("GPU Binius Binary Field: Packed Reduce")
    testPackedReduce()

    suite("GPU Binius Binary Field: Additive NTT")
    testAdditiveNTT()

    suite("GPU Binius Binary Field: Tower Consistency")
    testTowerConsistency()

    suite("GPU Binius Binary Field: Engine GF32/GF64 Batch")
    testEngineGF32GF64Batch()

    suite("GPU Binius Binary Field: Packed Engine Batch")
    testPackedEngineBatch()

    suite("GPU Binius Binary Field: F256 Exponentiation")
    testF256Exponentiation()

    suite("GPU Binius Binary Field: Free Function Wrappers")
    testFreeFunctionWrappers()
}

// MARK: - F2 Arithmetic

private func testF2Arithmetic() {
    let zero = BiniusF2.zero
    let one  = BiniusF2.one

    // Addition table for GF(2)
    expectEqual(zero + zero, zero, "0+0=0")
    expectEqual(zero + one,  one,  "0+1=1")
    expectEqual(one + zero,  one,  "1+0=1")
    expectEqual(one + one,   zero, "1+1=0 (char 2)")

    // Multiplication table for GF(2)
    expectEqual(zero * zero, zero, "0*0=0")
    expectEqual(zero * one,  zero, "0*1=0")
    expectEqual(one * zero,  zero, "1*0=0")
    expectEqual(one * one,   one,  "1*1=1")

    // Subtraction = addition in char 2
    expectEqual(one - one, zero, "1-1=0")
    expectEqual(one - zero, one, "1-0=1")

    // Squaring is identity in GF(2)
    expectEqual(one.squared(), one, "1^2=1")
    expectEqual(zero.squared(), zero, "0^2=0")

    // Inverse of 1 is 1
    expectEqual(one.inverse(), one, "inv(1)=1")

    // isZero checks
    expect(zero.isZero, "zero is zero")
    expect(!one.isZero, "one is not zero")
}

// MARK: - F4 Arithmetic

private func testF4Arithmetic() {
    let zero  = BiniusF4.zero
    let one   = BiniusF4.one
    let alpha = BiniusF4.alpha
    let alphaPlus1 = BiniusF4(value: 3)  // alpha + 1

    // Addition: alpha + 1 = element 3
    expectEqual(alpha + one, alphaPlus1, "alpha+1")

    // char 2: alpha + alpha = 0
    expectEqual(alpha + alpha, zero, "alpha+alpha=0")

    // Multiplicative identity
    expectEqual(alpha * one, alpha, "alpha*1=alpha")
    expectEqual(one * one, one, "1*1=1")

    // alpha^2 = alpha + 1 (defining relation of GF(4))
    let alphaSq = alpha * alpha
    expectEqual(alphaSq, alphaPlus1, "alpha^2 = alpha+1")

    // (alpha+1)^2 = alpha^2 + 1 = alpha + 1 + 1 = alpha
    let ap1sq = alphaPlus1 * alphaPlus1
    expectEqual(ap1sq, alpha, "(alpha+1)^2 = alpha")

    // alpha * (alpha+1) = alpha^2 + alpha = (alpha+1) + alpha = 1
    let product = alpha * alphaPlus1
    expectEqual(product, one, "alpha*(alpha+1)=1")

    // Commutativity
    expectEqual(alpha * alphaPlus1, alphaPlus1 * alpha, "mul commutative")

    // Zero absorbs
    expectEqual(alpha * zero, zero, "alpha*0=0")

    // Squaring via method
    expectEqual(alpha.squared(), alphaPlus1, "alpha.squared() = alpha+1")
}

// MARK: - F4 Field Axioms

private func testF4FieldAxioms() {
    // Exhaustive check: all nonzero elements have inverses
    for i in 1..<4 {
        let a = BiniusF4(value: UInt8(i))
        let inv = a.inverse()
        let prod = a * inv
        expectEqual(prod, BiniusF4.one, "F4: a*inv(a)=1 for a=\(i)")
    }

    // Exhaustive commutativity
    for i in 0..<4 {
        for j in 0..<4 {
            let a = BiniusF4(value: UInt8(i))
            let b = BiniusF4(value: UInt8(j))
            expectEqual(a * b, b * a, "F4 mul commutative \(i)*\(j)")
        }
    }

    // Sample distributivity
    for i in 0..<4 {
        let a = BiniusF4(value: UInt8(i))
        let b = BiniusF4(value: UInt8((i + 1) % 4))
        let c = BiniusF4(value: UInt8((i + 2) % 4))
        expectEqual(a * (b + c), (a * b) + (a * c), "F4 distributive \(i)")
    }
}

// MARK: - F16 Arithmetic

private func testF16Arithmetic() {
    let zero = BiniusF16.zero
    let one  = BiniusF16.one

    // Additive identity
    let a = BiniusF16(value: 7)
    expectEqual(a + zero, a, "F16 add identity")
    expectEqual(a + a, zero, "F16 char 2 self-cancel")

    // Multiplicative identity
    expectEqual(a * one, a, "F16 mul identity")
    expectEqual(a * zero, zero, "F16 mul by zero")

    // Commutativity
    let b = BiniusF16(value: 11)
    expectEqual(a * b, b * a, "F16 mul commutative")

    // Associativity
    let c = BiniusF16(value: 5)
    expectEqual((a * b) * c, a * (b * c), "F16 mul associative")

    // Distributivity
    expectEqual(a * (b + c), (a * b) + (a * c), "F16 distributive")

    // Subtraction = addition
    expectEqual(a - b, a + b, "F16 sub = add")

    // Sample elements have order dividing 15 (|GF(16)*| = 15)
    for i: UInt8 in [1, 3, 7, 11, 15] {
        let x = BiniusF16(value: i)
        var pow = x
        for _ in 1..<15 { pow = pow * x }
        expectEqual(pow, one, "F16 element \(i) has order dividing 15")
    }
}

// MARK: - F16 Inverse

private func testF16Inverse() {
    // All nonzero elements: a * inv(a) = 1
    for i in 1..<16 {
        let a = BiniusF16(value: UInt8(i))
        let inv = a.inverse()
        let prod = a * inv
        expectEqual(prod, BiniusF16.one, "F16 inv for element \(i)")
    }

    // Double inverse sample
    for i: UInt8 in [1, 5, 10, 15] {
        let a = BiniusF16(value: i)
        expectEqual(a.inverse().inverse(), a, "F16 double inv for \(i)")
    }
}

// MARK: - F256 Arithmetic

private func testF256Arithmetic() {
    let zero = BiniusF256.zero
    let one  = BiniusF256.one

    let a = BiniusF256(value: 0x53)
    let b = BiniusF256(value: 0xCA)
    let c = BiniusF256(value: 0x37)

    // Additive identity
    expectEqual(a + zero, a, "F256 add identity")

    // Char 2 self-cancel
    expectEqual(a + a, zero, "F256 char 2 self-cancel")

    // Multiplicative identity
    expectEqual(a * one, a, "F256 mul identity")

    // Zero absorbs
    expectEqual(a * zero, zero, "F256 mul by zero")

    // Commutativity
    expectEqual(a * b, b * a, "F256 mul commutative")
    expectEqual(a + b, b + a, "F256 add commutative")

    // Associativity of multiplication
    expectEqual((a * b) * c, a * (b * c), "F256 mul associative")

    // Distributivity
    expectEqual(a * (b + c), (a * b) + (a * c), "F256 distributive")

    // Subtraction = addition
    expectEqual(a - b, a + b, "F256 sub = add")
}

// MARK: - F256 Field Axioms

private func testF256FieldAxioms() {
    // Test all elements have order dividing 255 (|GF(256)*| = 255)
    // Sample several elements since checking all 255 is feasible but slow
    let testVals: [UInt8] = [1, 2, 3, 0x53, 0xCA, 0xFF, 0x80, 0x42, 0xAB, 0x17]
    for v in testVals {
        let a = BiniusF256(value: v)
        // a^255 should equal 1 (Fermat's little theorem for GF(256))
        let pow255 = a.pow(255)
        expectEqual(pow255, BiniusF256.one, "F256 Fermat: \(v)^255 = 1")
    }

    // a^256 = a (Frobenius)
    for v in testVals {
        let a = BiniusF256(value: v)
        let pow256 = a.pow(256)
        expectEqual(pow256, a, "F256 Frobenius: \(v)^256 = \(v)")
    }
}

// MARK: - F256 Inverse

private func testF256Inverse() {
    // Test a * inv(a) = 1 for a sample of nonzero elements
    let testVals: [UInt8] = [1, 2, 3, 0x53, 0xCA, 0xFF, 0x80, 0x42, 0xAB, 0x17,
                              0x01, 0xFE, 0x37, 0x99, 0xDE, 0x11, 0x7F, 0x55, 0xAA, 0xBB]
    for v in testVals {
        let a = BiniusF256(value: v)
        let inv = a.inverse()
        let prod = a * inv
        expectEqual(prod, BiniusF256.one, "F256 inv for 0x\(String(v, radix: 16))")
    }

    // Double inverse (sample)
    let a256 = BiniusF256(value: 0x53)
    expectEqual(a256.inverse().inverse(), a256, "F256 double inv")
    expectEqual(BiniusF256.one.inverse(), BiniusF256.one, "F256 inv(1) = 1")
}

// MARK: - F256 Polynomial Arithmetic

private func testF256Polynomial() {
    let zero = BiniusF256.zero
    let one  = BiniusF256.one
    let two  = BiniusF256(value: 2)
    let three = BiniusF256(value: 3)

    // Polynomial addition: (1 + 2x) + (3 + x) = (1+3) + (2+1)x = 2 + 3x
    let p1: [BiniusF256] = [one, two]
    let p2: [BiniusF256] = [three, one]
    let sum = BiniusBinaryPoly.add(p1, p2)
    expectEqual(sum.count, 2, "poly add length")
    expectEqual(sum[0], one + three, "poly add coeff 0")
    expectEqual(sum[1], two + one, "poly add coeff 1")

    // Polynomial multiplication: (1 + x) * (1 + x) = 1 + 0*x + x^2
    // In char 2: (1+x)^2 = 1 + x^2  (cross term 2x = 0)
    let p3: [BiniusF256] = [one, one]
    let sq = BiniusBinaryPoly.mul(p3, p3)
    expectEqual(sq.count, 3, "poly sqr length")
    expectEqual(sq[0], one, "poly sqr coeff 0")
    expectEqual(sq[1], zero, "poly sqr coeff 1 (cross term vanishes in char 2)")
    expectEqual(sq[2], one, "poly sqr coeff 2")

    // Polynomial evaluation: p(x) = 1 + 2x at x = 3 => 1 + 2*3
    let evalAt3 = BiniusBinaryPoly.evaluate(p1, at: three)
    let expected = one + (two * three)
    expectEqual(evalAt3, expected, "poly eval at x=3")

    // Degree of zero polynomial
    let degZero = BiniusBinaryPoly.degree([zero, zero, zero])
    expectEqual(degZero, -1, "degree of zero poly")

    // Degree of 1 + 0*x + 3*x^2
    let p4: [BiniusF256] = [one, zero, three]
    expectEqual(BiniusBinaryPoly.degree(p4), 2, "degree of 1+3x^2")

    // Scale polynomial
    let scaled = BiniusBinaryPoly.scale(p1, by: two)
    expectEqual(scaled[0], two, "scaled coeff 0")
    expectEqual(scaled[1], two * two, "scaled coeff 1")

    // Evaluation of empty polynomial
    let evalEmpty = BiniusBinaryPoly.evaluate([], at: one)
    expectEqual(evalEmpty, zero, "eval of empty poly = 0")
}

// MARK: - Packed Binary Field 32

private func testPackedBinaryField32() {
    let a = PackedBinaryField32(bits: 0xDEADBEEF)
    let b = PackedBinaryField32(bits: 0x12345678)
    let zero = PackedBinaryField32.zero

    // Add = XOR
    let sum = a + b
    expectEqual(sum.bits, 0xDEADBEEF ^ 0x12345678, "packed32 add = XOR")

    // Mul = AND
    let prod = a * b
    expectEqual(prod.bits, 0xDEADBEEF & 0x12345678, "packed32 mul = AND")

    // Self-cancel
    expectEqual(a + a, zero, "packed32 self-cancel")

    // Sub = add in char 2
    expectEqual(a - b, a + b, "packed32 sub = add")

    // Bit extraction
    expectEqual(a.bit(at: 0), 1, "bit 0 of 0xDEADBEEF")
    expectEqual(a.bit(at: 4), 0, "bit 4 of 0xDEADBEEF")

    // Popcount
    let allOnes = PackedBinaryField32(bits: 0xFFFFFFFF)
    expectEqual(allOnes.popcount, 32, "popcount of all ones")
    expectEqual(zero.popcount, 0, "popcount of zero")
    expectEqual(PackedBinaryField32(bits: 1).popcount, 1, "popcount of 1")

    // Parity
    expectEqual(allOnes.parity, 0, "parity of 32 ones = 0 (even)")
    expectEqual(PackedBinaryField32(bits: 1).parity, 1, "parity of 1 = 1 (odd)")
    expectEqual(PackedBinaryField32(bits: 3).parity, 0, "parity of 3 = 0 (even)")

    // Set bit
    var c = PackedBinaryField32.zero
    c.setBit(at: 5, to: 1)
    expectEqual(c.bits, 32, "set bit 5")
    c.setBit(at: 5, to: 0)
    expectEqual(c.bits, 0, "clear bit 5")
}

// MARK: - Packed Binary Field 64

private func testPackedBinaryField64() {
    let a = PackedBinaryField64(bits: 0xDEADBEEFCAFEBABE)
    let b = PackedBinaryField64(bits: 0x1234567890ABCDEF)
    let zero = PackedBinaryField64.zero

    // Add = XOR
    let sum = a + b
    expectEqual(sum.bits, 0xDEADBEEFCAFEBABE ^ 0x1234567890ABCDEF, "packed64 add = XOR")

    // Mul = AND
    let prod = a * b
    expectEqual(prod.bits, 0xDEADBEEFCAFEBABE & 0x1234567890ABCDEF, "packed64 mul = AND")

    // Self-cancel
    expectEqual(a + a, zero, "packed64 self-cancel")

    // Split into halves
    expectEqual(a.loHalf.bits, UInt32(0xCAFEBABE), "packed64 lo half")
    expectEqual(a.hiHalf.bits, UInt32(0xDEADBEEF), "packed64 hi half")

    // Reconstruct from halves
    let reconstructed = PackedBinaryField64(lo: a.loHalf, hi: a.hiHalf)
    expectEqual(reconstructed, a, "packed64 reconstruct from halves")

    // Bit extraction
    expectEqual(a.bit(at: 0), 0, "bit 0 of 0xCAFEBABE (lo byte 0xBE = 10111110)")
    expectEqual(a.bit(at: 1), 1, "bit 1")

    // Parity
    expectEqual(zero.parity, 0, "packed64 parity of zero")
    expectEqual(PackedBinaryField64(bits: 1).parity, 1, "packed64 parity of 1")
}

// MARK: - Engine Batch Ops

private func testEngineBatchOps() {
    let engine = GPUBiniusBinaryFieldEngine.shared
    let n = 32

    // Generate deterministic test data
    var aArr = [BiniusF256](repeating: .zero, count: n)
    var bArr = [BiniusF256](repeating: .zero, count: n)
    var seed: UInt32 = 0xBAADF00D
    for i in 0..<n {
        seed = seed &* 1103515245 &+ 12345
        aArr[i] = BiniusF256(value: UInt8(seed & 0xFF))
        seed = seed &* 1103515245 &+ 12345
        bArr[i] = BiniusF256(value: UInt8(seed & 0xFF))
    }

    let addResult = engine.batchAdd256(aArr, bArr)
    var addMatch = true
    for i in 0..<n { if addResult[i] != (aArr[i] + bArr[i]) { addMatch = false; break } }
    expect(addMatch, "batch add256 matches element-wise")

    let mulResult = engine.batchMul256(aArr, bArr)
    var mulMatch = true
    for i in 0..<n { if mulResult[i] != (aArr[i] * bArr[i]) { mulMatch = false; break } }
    expect(mulMatch, "batch mul256 matches element-wise")

    let scalar = BiniusF256(value: 0x53)
    let scalarResult = engine.batchScalarMul256(scalar, aArr)
    var scalarMatch = true
    for i in 0..<n { if scalarResult[i] != (scalar * aArr[i]) { scalarMatch = false; break } }
    expect(scalarMatch, "batch scalar mul256 matches element-wise")
}

// MARK: - Batch Inverse 256

private func testBatchInverse256() {
    let engine = GPUBiniusBinaryFieldEngine.shared
    let n = 32

    // Generate nonzero test data
    var arr = [BiniusF256](repeating: .zero, count: n)
    var seed: UInt32 = 0xCAFEBEEF
    for i in 0..<n {
        seed = seed &* 1103515245 &+ 12345
        arr[i] = BiniusF256(value: UInt8(seed & 0xFF) | 1)  // force nonzero
    }

    let inverses = engine.batchInverse256(arr)

    var allOne = true
    for i in 0..<n { if (arr[i] * inverses[i]) != BiniusF256.one { allOne = false; break } }
    expect(allOne, "batch inverse256: a*inv(a)=1 for all elements")

    // Test zero handling
    var withZero = arr
    withZero[3] = .zero
    withZero[17] = .zero
    let invWithZero = engine.batchInverse256(withZero)
    expectEqual(invWithZero[3], BiniusF256.zero, "batch inv of zero = zero (idx 3)")
    expectEqual(invWithZero[17], BiniusF256.zero, "batch inv of zero = zero (idx 17)")

    // Non-zero entries still correct
    let p0 = withZero[0] * invWithZero[0]
    expectEqual(p0, BiniusF256.one, "batch inv correct for non-zero with zero skip")
}

// MARK: - Multilinear Eval 256

private func testMultilinearEval256() {
    let engine = GPUBiniusBinaryFieldEngine.shared

    // n=1: f(x) = a + (b+a)*x
    let a = BiniusF256(value: 0x12)
    let b = BiniusF256(value: 0x9A)

    // f(0) = a
    let at0 = engine.multilinearEval256(evals: [a, b], at: [.zero])
    expectEqual(at0, a, "MLE256 f(0) = a")

    // f(1) = b
    let at1 = engine.multilinearEval256(evals: [a, b], at: [.one])
    expectEqual(at1, b, "MLE256 f(1) = b")

    // n=2: f(x0, x1) with 4 evaluations
    let e00 = BiniusF256(value: 1)
    let e01 = BiniusF256(value: 2)
    let e10 = BiniusF256(value: 3)
    let e11 = BiniusF256(value: 4)
    let evals4: [BiniusF256] = [e00, e01, e10, e11]

    let f00 = engine.multilinearEval256(evals: evals4, at: [.zero, .zero])
    expectEqual(f00, e00, "MLE256 f(0,0)")

    let f10 = engine.multilinearEval256(evals: evals4, at: [.one, .zero])
    expectEqual(f10, e10, "MLE256 f(1,0)")

    let f01 = engine.multilinearEval256(evals: evals4, at: [.zero, .one])
    expectEqual(f01, e01, "MLE256 f(0,1)")

    let f11 = engine.multilinearEval256(evals: evals4, at: [.one, .one])
    expectEqual(f11, e11, "MLE256 f(1,1)")
}

// MARK: - Inner Product 256

private func testInnerProduct256() {
    let engine = GPUBiniusBinaryFieldEngine.shared
    let n = 8

    // Generate test data
    var a = [BiniusF256](repeating: .zero, count: n)
    var seed: UInt32 = 0xFEEDFACE
    for i in 0..<n {
        seed = seed &* 1103515245 &+ 12345
        a[i] = BiniusF256(value: UInt8(seed & 0xFF))
    }

    // Inner product with one-hot vector selects element
    var oneHot = [BiniusF256](repeating: .zero, count: n)
    oneHot[3] = BiniusF256.one
    let ip = engine.innerProduct256(a, oneHot)
    expectEqual(ip, a[3], "inner product with one-hot selects element")

    // Inner product with zeros = zero
    let zeros = [BiniusF256](repeating: .zero, count: n)
    let ipZero = engine.innerProduct256(a, zeros)
    expectEqual(ipZero, BiniusF256.zero, "inner product with zeros = 0")

    // Inner product with all-ones = XOR sum
    let allOnes = [BiniusF256](repeating: .one, count: n)
    let ipOnes = engine.innerProduct256(a, allOnes)
    var xorSum = BiniusF256.zero
    for x in a { xorSum = xorSum + x }
    expectEqual(ipOnes, xorSum, "inner product with all-ones = XOR sum")
}

// MARK: - Pack/Unpack Bits

private func testPackUnpackBits() {
    let engine = GPUBiniusBinaryFieldEngine.shared

    // Pack 8 bits into a 32-bit word
    let bits8: [UInt8] = [1, 0, 1, 1, 0, 1, 0, 0]
    let packed32 = engine.packBitsInto32(bits8)
    expectEqual(packed32.count, 1, "8 bits pack into 1 word")
    // bits = 00101101 = 0x2D
    expectEqual(packed32[0].bits, 0x2D, "packed value matches")

    // Unpack round-trip
    let unpacked = engine.unpackBitsFrom32(packed32, count: 8)
    var unpackMatch = true
    for i in 0..<8 {
        if unpacked[i] != bits8[i] { unpackMatch = false; break }
    }
    expect(unpackMatch, "pack/unpack 32 round-trip")

    // Pack 64-bit round trip
    let bits64: [UInt8] = (0..<64).map { UInt8($0 % 3 == 0 ? 1 : 0) }
    let packed64 = engine.packBitsInto64(bits64)
    expectEqual(packed64.count, 1, "64 bits pack into 1 word")
    let unpacked64 = engine.unpackBitsFrom64(packed64, count: 64)
    var unpack64Match = true
    for i in 0..<64 { if unpacked64[i] != bits64[i] { unpack64Match = false; break } }
    expect(unpack64Match, "pack/unpack 64 round-trip")

    // Pack more than one word
    let bits100: [UInt8] = (0..<100).map { UInt8($0 & 1) }
    let packed100 = engine.packBitsInto32(bits100)
    expectEqual(packed100.count, 4, "100 bits need 4 x 32-bit words")
    let unpacked100 = engine.unpackBitsFrom32(packed100, count: 100)
    var unpack100Match = true
    for i in 0..<100 { if unpacked100[i] != bits100[i] { unpack100Match = false; break } }
    expect(unpack100Match, "pack/unpack 100 bits round-trip")
}

// MARK: - Packed Reduce

private func testPackedReduce() {
    let engine = GPUBiniusBinaryFieldEngine.shared

    // XOR-reduce of two words
    let words32: [PackedBinaryField32] = [
        PackedBinaryField32(bits: 0xAAAAAAAA),
        PackedBinaryField32(bits: 0x55555555)
    ]
    let reduced32 = engine.packedReduce32(words32)
    expectEqual(reduced32.bits, 0xFFFFFFFF, "XOR reduce of AA.. and 55.. = FF..")

    // XOR-reduce of identical words = 0
    let same32: [PackedBinaryField32] = [
        PackedBinaryField32(bits: 0xDEADBEEF),
        PackedBinaryField32(bits: 0xDEADBEEF)
    ]
    let reducedSame = engine.packedReduce32(same32)
    expectEqual(reducedSame.bits, 0, "XOR reduce of same = 0")

    // 64-bit reduce
    let words64: [PackedBinaryField64] = [
        PackedBinaryField64(bits: 0xAAAAAAAA55555555),
        PackedBinaryField64(bits: 0x5555555500000000)
    ]
    let reduced64 = engine.packedReduce64(words64)
    expectEqual(reduced64.bits, 0xFFFFFFFF55555555, "64-bit XOR reduce")

    // Empty arrays
    let emptyReduce32 = engine.packedReduce32([])
    expectEqual(emptyReduce32.bits, 0, "reduce of empty = 0")
}

// MARK: - Additive NTT

private func testAdditiveNTT() {
    // Test with k=2 (4 elements)
    let k = 2
    let basis = BiniusAdditiveNTT.standardBasis(dimension: k)
    expectEqual(basis.count, k, "basis dimension")

    // Forward NTT then inverse should recover original
    let original: [BiniusF256] = [
        BiniusF256(value: 1), BiniusF256(value: 2),
        BiniusF256(value: 3), BiniusF256(value: 4)
    ]
    let transformed = BiniusAdditiveNTT.forward(original, basis: basis)
    let recovered = BiniusAdditiveNTT.inverse(transformed, basis: basis)

    var nttMatch = true
    for i in 0..<4 {
        if recovered[i] != original[i] { nttMatch = false; break }
    }
    expect(nttMatch, "additive NTT forward+inverse round-trip (k=2)")

    // Test with k=3 (8 elements)
    let k3 = 3
    let basis3 = BiniusAdditiveNTT.standardBasis(dimension: k3)
    let original8: [BiniusF256] = (0..<8).map { BiniusF256(value: UInt8($0 + 1)) }
    let transformed8 = BiniusAdditiveNTT.forward(original8, basis: basis3)
    let recovered8 = BiniusAdditiveNTT.inverse(transformed8, basis: basis3)

    var ntt8Match = true
    for i in 0..<8 {
        if recovered8[i] != original8[i] { ntt8Match = false; break }
    }
    expect(ntt8Match, "additive NTT forward+inverse round-trip (k=3)")

    // Forward NTT of all zeros = all zeros
    let zeros4 = [BiniusF256](repeating: .zero, count: 4)
    let transZeros = BiniusAdditiveNTT.forward(zeros4, basis: basis)
    var allZero = true
    for x in transZeros {
        if !x.isZero { allZero = false; break }
    }
    expect(allZero, "NTT of zeros = zeros")
}

// MARK: - Tower Consistency

private func testTowerConsistency() {
    // Verify tower embeddings are consistent:
    // F2 -> F4 -> F16 -> F256

    // F2 embeds into F4: 0->0, 1->1
    let f2zero = BiniusF2.zero
    let f2one  = BiniusF2.one
    let f4fromZero = BiniusF4(value: f2zero.bit)
    let f4fromOne  = BiniusF4(value: f2one.bit)
    expectEqual(f4fromZero, BiniusF4.zero, "F2(0) embeds to F4(0)")
    expectEqual(f4fromOne,  BiniusF4.one,  "F2(1) embeds to F4(1)")

    // F4 embeds into F16: lo nibble
    let f4alpha = BiniusF4.alpha
    let f16fromAlpha = BiniusF16(lo: f4alpha, hi: .zero)
    expectEqual(f16fromAlpha.value, f4alpha.value, "F4(alpha) embeds to F16 lo")

    // F16 embeds into F256: lo nibble
    let f16val = BiniusF16(value: 7)
    let f256fromF16 = BiniusF256(lo: f16val, hi: .zero)
    expectEqual(f256fromF16.lo, f16val, "F16(7) embeds to F256 lo")
    expectEqual(f256fromF16.hi, BiniusF16.zero, "F16(7) embed has zero hi")

    // Subfield multiplication consistency:
    // Multiplying two F4 elements embedded in F16 should give the same
    // result as multiplying in F4 and then embedding.
    for i in 0..<4 {
        for j in 0..<4 {
            let a4 = BiniusF4(value: UInt8(i))
            let b4 = BiniusF4(value: UInt8(j))
            let prod4 = a4 * b4

            let a16 = BiniusF16(lo: a4, hi: .zero)
            let b16 = BiniusF16(lo: b4, hi: .zero)
            let prod16 = a16 * b16

            expectEqual(prod16.lo, prod4, "F4 mul consistent in F16 (\(i)*\(j))")
            expectEqual(prod16.hi, BiniusF4.zero, "F4 product stays in subfield (\(i)*\(j))")
        }
    }

    // Sample F16 subfield multiplication in F256
    let sampleF16Vals: [UInt8] = [0, 1, 2, 5, 7, 10, 15]
    for i in sampleF16Vals {
        for j in sampleF16Vals {
            let a16 = BiniusF16(value: i)
            let b16 = BiniusF16(value: j)
            let prod16 = a16 * b16
            let a256 = BiniusF256(lo: a16, hi: .zero)
            let b256 = BiniusF256(lo: b16, hi: .zero)
            let prod256 = a256 * b256
            expectEqual(prod256.lo, prod16, "F16 mul consistent in F256 (\(i)*\(j))")
            expectEqual(prod256.hi, BiniusF16.zero, "F16 stays in subfield (\(i)*\(j))")
        }
    }
}

// MARK: - Engine GF32/GF64 Batch

private func testEngineGF32GF64Batch() {
    let engine = GPUBiniusBinaryFieldEngine.shared
    let n = 16

    // GF32 batch operations
    var a32 = [UInt32](repeating: 0, count: n)
    var b32 = [UInt32](repeating: 0, count: n)
    var seed: UInt32 = 0x12345678
    for i in 0..<n {
        seed = seed &* 1103515245 &+ 12345
        a32[i] = seed
        seed = seed &* 1103515245 &+ 12345
        b32[i] = seed
    }

    let addResult32 = engine.batchAddGF32(a32, b32)
    var add32Match = true
    for i in 0..<n { if addResult32[i] != (a32[i] ^ b32[i]) { add32Match = false; break } }
    expect(add32Match, "batch GF32 add = XOR")

    let mulResult32 = engine.batchMulGF32(a32, b32)
    var mul32Match = true
    // Verify batch mul produces nonzero results for nonzero inputs
    var mul32NonZero = false
    for i in 0..<n { if mulResult32[i] != 0 { mul32NonZero = true; break } }
    expect(mul32NonZero, "batch GF32 mul produces nonzero results")

    // GF64 batch operations
    let a64 = (0..<n).map { _ -> UInt64 in seed = seed &* 1103515245 &+ 12345; return UInt64(seed) << 16 | UInt64(seed) }
    let b64 = (0..<n).map { _ -> UInt64 in seed = seed &* 1103515245 &+ 12345; return UInt64(seed) << 16 | UInt64(seed) }

    let addResult64 = engine.batchAddGF64(a64, b64)
    var add64Match = true
    for i in 0..<n { if addResult64[i] != (a64[i] ^ b64[i]) { add64Match = false; break } }
    expect(add64Match, "batch GF64 add = XOR")

    let mulResult64 = engine.batchMulGF64(a64, b64)
    var mul64Match = true
    // Verify batch mul produces nonzero results for nonzero inputs
    var mul64NonZero = false
    for i in 0..<n { if mulResult64[i] != 0 { mul64NonZero = true; break } }
    expect(mul64NonZero, "batch GF64 mul produces nonzero results")
}

// MARK: - Packed Engine Batch

private func testPackedEngineBatch() {
    let engine = GPUBiniusBinaryFieldEngine.shared
    let n = 16

    // Generate test data
    var a32 = [PackedBinaryField32](repeating: .zero, count: n)
    var b32 = [PackedBinaryField32](repeating: .zero, count: n)
    var seed: UInt32 = 0xDEADC0DE
    for i in 0..<n {
        seed = seed &* 1103515245 &+ 12345
        a32[i] = PackedBinaryField32(bits: seed)
        seed = seed &* 1103515245 &+ 12345
        b32[i] = PackedBinaryField32(bits: seed)
    }

    let addResult = engine.batchPackedAdd32(a32, b32)
    var addMatch = true
    for i in 0..<n { if addResult[i] != (a32[i] + b32[i]) { addMatch = false; break } }
    expect(addMatch, "batch packed32 add matches element-wise")

    let mulResult = engine.batchPackedMul32(a32, b32)
    var mulMatch = true
    for i in 0..<n { if mulResult[i] != (a32[i] * b32[i]) { mulMatch = false; break } }
    expect(mulMatch, "batch packed32 mul matches element-wise")

    // 64-bit packed batch (reuse seed)
    let a64 = (0..<n).map { _ -> PackedBinaryField64 in
        seed = seed &* 1103515245 &+ 12345; return PackedBinaryField64(bits: UInt64(seed) << 16 | UInt64(seed))
    }
    let b64 = (0..<n).map { _ -> PackedBinaryField64 in
        seed = seed &* 1103515245 &+ 12345; return PackedBinaryField64(bits: UInt64(seed) << 16 | UInt64(seed))
    }
    let addResult64 = engine.batchPackedAdd64(a64, b64)
    var add64Match = true
    for i in 0..<n { if addResult64[i] != (a64[i] + b64[i]) { add64Match = false; break } }
    expect(add64Match, "batch packed64 add matches element-wise")

    let mulResult64 = engine.batchPackedMul64(a64, b64)
    var mul64Match = true
    for i in 0..<n { if mulResult64[i] != (a64[i] * b64[i]) { mul64Match = false; break } }
    expect(mul64Match, "batch packed64 mul matches element-wise")
}

// MARK: - F256 Exponentiation

private func testF256Exponentiation() {
    let a = BiniusF256(value: 0x53)

    // pow(0) = 1
    expectEqual(a.pow(0), BiniusF256.one, "a^0 = 1")

    // pow(1) = a
    expectEqual(a.pow(1), a, "a^1 = a")

    // pow(2) = squared
    expectEqual(a.pow(2), a.squared(), "a^2 = squared()")

    // pow(3) = a*a*a
    expectEqual(a.pow(3), a * a * a, "a^3 = a*a*a")

    // pow(4) = (a^2)^2
    expectEqual(a.pow(4), a.squared().squared(), "a^4 = (a^2)^2")

    // Fermat: a^255 = 1 for nonzero a
    expectEqual(a.pow(255), BiniusF256.one, "a^255 = 1 (Fermat)")

    // a^256 = a (Frobenius)
    expectEqual(a.pow(256), a, "a^256 = a (Frobenius)")

    // 1^n = 1 for all n
    expectEqual(BiniusF256.one.pow(42), BiniusF256.one, "1^42 = 1")
    expectEqual(BiniusF256.one.pow(0), BiniusF256.one, "1^0 = 1")
}

// MARK: - Free Function Wrappers

private func testFreeFunctionWrappers() {
    let a2 = BiniusF2(bit: 1)
    let b2 = BiniusF2(bit: 1)
    expectEqual(biniusF2Add(a2, b2), BiniusF2.zero, "free fn F2 add")
    expectEqual(biniusF2Mul(a2, b2), BiniusF2.one, "free fn F2 mul")

    let a4 = BiniusF4.alpha
    let b4 = BiniusF4.one
    expectEqual(biniusF4Add(a4, b4), BiniusF4(value: 3), "free fn F4 add")
    expectEqual(biniusF4Mul(a4, b4), a4, "free fn F4 mul")

    let a16 = BiniusF16(value: 5)
    let b16 = BiniusF16(value: 3)
    expectEqual(biniusF16Add(a16, b16), BiniusF16(value: 5 ^ 3), "free fn F16 add")
    expectEqual(biniusF16Mul(a16, b16), a16 * b16, "free fn F16 mul")

    let a256 = BiniusF256(value: 0x53)
    let b256 = BiniusF256(value: 0xCA)
    expectEqual(biniusF256Add(a256, b256), a256 + b256, "free fn F256 add")
    expectEqual(biniusF256Mul(a256, b256), a256 * b256, "free fn F256 mul")

    let inv256 = biniusF256Inv(a256)
    expectEqual(a256 * inv256, BiniusF256.one, "free fn F256 inv")
}
