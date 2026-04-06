import zkMetal
import Foundation

public func runGPUBiniusPolyTests() {
    suite("Binius Poly128: Construction and Basics")
    testPoly128Basics()

    suite("Binius Poly128: Addition")
    testPoly128Add()

    suite("Binius Poly128: Multiplication")
    testPoly128Mul()

    suite("Binius Poly128: Evaluation (Horner)")
    testPoly128Eval()

    suite("Binius Poly128: Division with Remainder")
    testPoly128DivMod()

    suite("Binius Poly128: From Roots")
    testPoly128FromRoots()

    suite("Binius Poly128: Derivative in Char 2")
    testPoly128Derivative()

    suite("Binius Poly128: Scalar Multiply")
    testPoly128ScalarMul()

    suite("Binius Multilinear128: Evaluation")
    testMultilinearEval128()

    suite("Binius Multilinear128: Partial Evaluation")
    testMultilinearPartialEval()

    suite("Binius Multilinear128: Sumcheck Round")
    testMultilinearSumcheckRound()

    suite("Binius Multilinear128: Sum All")
    testMultilinearSumAll()

    suite("Binius Multilinear128: Tensor Product")
    testMultilinearTensorProduct()

    suite("Binius AdditiveFFT128: Forward-Inverse Roundtrip")
    testAdditiveFFT128Roundtrip()

    suite("Binius AdditiveFFT128: Linearity of Forward")
    testAdditiveFFT128Linearity()

    suite("Binius AdditiveFFT128: Canonical Basis")
    testCanonicalBasis()

    suite("Binius AdditiveFFT128: Primitive Basis")
    testPrimitiveBasis()

    suite("Binius Reed-Solomon128: Encode-Decode Roundtrip")
    testReedSolomonRoundtrip()

    suite("Binius Reed-Solomon128: Valid Codeword Check")
    testReedSolomonValidCodeword()

    suite("Binius Reed-Solomon128: FRI Fold")
    testReedSolomonFRIFold()

    suite("Binius Reed-Solomon128: Proximity Distance")
    testReedSolomonProximity()

    suite("Binius Sumcheck128: Prover State")
    testSumcheckProverState()

    suite("Binius Sumcheck128: Verify Round")
    testSumcheckVerifyRound()

    suite("Binius Sumcheck128: Full Protocol")
    testSumcheckFullProtocol()

    suite("Binius PackedGF2Poly: Basics")
    testPackedGF2PolyBasics()

    suite("Binius PackedGF2Poly: Multiplication")
    testPackedGF2PolyMul()

    suite("Binius PackedGF2Poly: DivMod")
    testPackedGF2PolyDivMod()

    suite("Binius PackedGF2Poly: GCD")
    testPackedGF2PolyGCD()

    suite("Binius PackedGF2Poly: Evaluation at Bit")
    testPackedGF2PolyEvalAtBit()

    suite("Binius Engine: Batch Multilinear Eval")
    testEngineBatchMultilinearEval()

    suite("Binius Engine: Batch Eval")
    testEngineBatchEval()

    suite("Binius Engine: Interpolation")
    testEngineInterpolation()

    suite("Binius Engine: FRI Fold Poly")
    testEngineFRIFoldPoly()

    suite("Binius Engine: Low Degree Extension")
    testEngineLDE()

    suite("Binius Engine: Vanishing Polynomial")
    testEngineVanishingPoly()

    suite("Binius Engine: Degree Bound Check")
    testEngineDegreeBound()

    suite("Binius Engine: Compose Polynomials")
    testEngineCompose()

    suite("Binius Engine: Linear Combination")
    testEngineLinearCombination()

    suite("Binius Engine: Batch Inner Products")
    testEngineBatchInnerProducts()

    suite("Binius Engine: Batch Forward FFT")
    testEngineBatchFFT()
}

// MARK: - Helpers

/// Create a deterministic BiniusTower128 element from a seed.
private func makeTower(_ seed: UInt32) -> BiniusTower128 {
    let w0 = seed &* 0xDEADBEEF
    let w1 = seed &* 0x12345678 ^ 0xCAFEBABE
    let w2 = seed &* 0x87654321 ^ 0x11223344
    let w3 = seed &* 0xAAAABBBB ^ 0x55667788
    return BiniusTower128(w0: w0, w1: w1, w2: w2, w3: w3)
}

/// Generate a vector of deterministic tower elements.
private func makeTowerVec(_ count: Int, seed: UInt32 = 1) -> [BiniusTower128] {
    (0..<count).map { makeTower(seed &+ UInt32($0)) }
}

// MARK: - PackedBinaryPoly128 Tests

private func testPoly128Basics() {
    // Zero polynomial
    let zero = PackedBinaryPoly128.zero
    expect(zero.isZero, "zero poly is zero")
    expectEqual(zero.degree, -1, "zero poly degree is -1")
    expectEqual(zero.count, 0, "zero poly has no coefficients")

    // Constant polynomial
    let c = BiniusTower128(w0: 42, w1: 0, w2: 0, w3: 0)
    let constPoly = PackedBinaryPoly128(constant: c)
    expectEqual(constPoly.degree, 0, "constant poly degree is 0")
    expectEqual(constPoly[0], c, "constant poly coefficient")
    expectEqual(constPoly[1], BiniusTower128.zero, "out of bounds gives zero")

    // Monomial
    let mono = PackedBinaryPoly128(monomial: BiniusTower128.one, degree: 3)
    expectEqual(mono.degree, 3, "monomial degree")
    expectEqual(mono[3], BiniusTower128.one, "monomial leading coeff")
    expectEqual(mono[0], BiniusTower128.zero, "monomial zero lower coeff")
    expectEqual(mono[2], BiniusTower128.zero, "monomial zero middle coeff")

    // Normalization
    var padded = PackedBinaryPoly128(coeffs: [c, .zero, .zero])
    expectEqual(padded.degree, 0, "degree ignores trailing zeros")
    padded.normalize()
    expectEqual(padded.count, 1, "normalize strips trailing zeros")
}

private func testPoly128Add() {
    let a = PackedBinaryPoly128(coeffs: [makeTower(1), makeTower(2), makeTower(3)])
    let b = PackedBinaryPoly128(coeffs: [makeTower(4), makeTower(5)])

    let sum = a + b
    // Addition is XOR
    expectEqual(sum[0], makeTower(1) + makeTower(4), "add coeff 0")
    expectEqual(sum[1], makeTower(2) + makeTower(5), "add coeff 1")
    expectEqual(sum[2], makeTower(3), "add coeff 2 (b has no X^2 term)")

    // a + a = 0 (char 2)
    let selfSum = a + a
    expect(selfSum.normalized().isZero, "a + a = 0 in char 2")

    // a - b = a + b (char 2)
    let diff = a - b
    expectEqual(diff[0], sum[0], "sub equals add coeff 0")
    expectEqual(diff[1], sum[1], "sub equals add coeff 1")
}

private func testPoly128Mul() {
    // (1 + X) * (1 + X) = 1 + X + X + X^2 = 1 + X^2 (char 2: X+X=0)
    let one = BiniusTower128.one
    let linPoly = PackedBinaryPoly128(coeffs: [one, one])  // 1 + X
    let sq = linPoly * linPoly
    expectEqual(sq.degree, 2, "square of linear has degree 2")
    expectEqual(sq[0], one, "constant term of (1+X)^2")
    expectEqual(sq[1], BiniusTower128.zero, "linear term of (1+X)^2 is 0 in char 2")
    expectEqual(sq[2], one, "quadratic term of (1+X)^2")

    // Multiplication by zero
    let z = linPoly * PackedBinaryPoly128.zero
    expect(z.normalized().isZero, "mul by zero gives zero")

    // Multiplication by one
    let constOne = PackedBinaryPoly128(constant: one)
    let prod = linPoly * constOne
    expectEqual(prod[0], one, "mul by 1 preserves coeff 0")
    expectEqual(prod[1], one, "mul by 1 preserves coeff 1")

    // Degree check: deg(a*b) = deg(a) + deg(b)
    let a = PackedBinaryPoly128(coeffs: [makeTower(1), makeTower(2), makeTower(3)])
    let b = PackedBinaryPoly128(coeffs: [makeTower(4), makeTower(5)])
    let ab = a * b
    expect(ab.degree <= a.degree + b.degree, "product degree <= sum of degrees")
}

private func testPoly128Eval() {
    // P(X) = c0 + c1*X + c2*X^2
    let c0 = BiniusTower128(w0: 3, w1: 0, w2: 0, w3: 0)
    let c1 = BiniusTower128(w0: 5, w1: 0, w2: 0, w3: 0)
    let c2 = BiniusTower128(w0: 7, w1: 0, w2: 0, w3: 0)
    let poly = PackedBinaryPoly128(coeffs: [c0, c1, c2])

    // P(0) = c0
    let evalZero = poly.evaluate(at: .zero)
    expectEqual(evalZero, c0, "P(0) = constant term")

    // P(1) = c0 + c1 + c2 (since 1^k = 1 for all k)
    let evalOne = poly.evaluate(at: .one)
    let expected = c0 + c1 + c2
    expectEqual(evalOne, expected, "P(1) = sum of coefficients")

    // Zero poly evaluates to zero everywhere
    let zeroEval = PackedBinaryPoly128.zero.evaluate(at: makeTower(42))
    expectEqual(zeroEval, BiniusTower128.zero, "zero poly evaluates to zero")

    // Constant poly evaluates to constant
    let constPoly = PackedBinaryPoly128(constant: c0)
    let constEval = constPoly.evaluate(at: makeTower(99))
    expectEqual(constEval, c0, "constant poly evaluates to constant")
}

private func testPoly128DivMod() {
    // Construct P = Q * D + R, then verify divmod recovers Q and R
    let d = PackedBinaryPoly128(coeffs: [makeTower(1), makeTower(2), .one])  // degree 2
    let q = PackedBinaryPoly128(coeffs: [makeTower(3), .one])  // degree 1
    let r = PackedBinaryPoly128(constant: makeTower(4))  // degree 0

    let p = (q * d) + r

    let (gotQ, gotR) = p.divmod(by: d)

    // Verify P = Q * D + R
    let reconstructed = (gotQ * d) + gotR
    // Check each coefficient
    let pNorm = p.normalized()
    let recNorm = reconstructed.normalized()
    expect(pNorm.degree == recNorm.degree, "divmod reconstruction preserves degree")
    for i in 0...max(pNorm.degree, 0) {
        expectEqual(pNorm[i], recNorm[i], "divmod reconstruction coeff \(i)")
    }

    // Remainder degree < divisor degree
    expect(gotR.degree < d.degree, "remainder degree < divisor degree")

    // Division of lower-degree by higher-degree
    let (q2, r2) = r.divmod(by: d)
    expect(q2.isZero, "dividing lower degree gives zero quotient")
    expectEqual(r2[0], r[0], "dividing lower degree gives self as remainder")

    // Mod operation
    let modResult = p.mod(by: d)
    expectEqual(modResult.degree, gotR.degree, "mod gives same degree as remainder")
}

private func testPoly128FromRoots() {
    // Build polynomial from roots and verify it vanishes at those roots
    let roots = [makeTower(1), makeTower(2), makeTower(3)]
    let poly = PackedBinaryPoly128.fromRoots(roots)

    expectEqual(poly.degree, 3, "polynomial from 3 roots has degree 3")

    // Verify vanishing at each root
    for (i, root) in roots.enumerated() {
        let val = poly.evaluate(at: root)
        expectEqual(val, BiniusTower128.zero, "polynomial vanishes at root \(i)")
    }

    // Empty roots => constant 1
    let emptyPoly = PackedBinaryPoly128.fromRoots([])
    expectEqual(emptyPoly.degree, 0, "empty roots gives degree 0")
    expectEqual(emptyPoly[0], BiniusTower128.one, "empty roots gives constant 1")

    // Single root: X + r
    let singleRoot = makeTower(42)
    let singlePoly = PackedBinaryPoly128.fromRoots([singleRoot])
    expectEqual(singlePoly.degree, 1, "single root gives degree 1")
    let evalSingle = singlePoly.evaluate(at: singleRoot)
    expectEqual(evalSingle, BiniusTower128.zero, "vanishes at single root")
}

private func testPoly128Derivative() {
    // d/dX(c0 + c1*X + c2*X^2 + c3*X^3) in char 2:
    // = c1 + 0 + 3*c3*X^2 = c1 + c3*X^2 (since 2=0, 3=1 mod 2)
    let c0 = makeTower(1)
    let c1 = makeTower(2)
    let c2 = makeTower(3)
    let c3 = makeTower(4)
    let poly = PackedBinaryPoly128(coeffs: [c0, c1, c2, c3])
    let deriv = poly.derivative()

    // The derivative should have c1 at X^0 and c3 at X^2
    expectEqual(deriv[0], c1, "derivative constant = c1")
    expectEqual(deriv[1], BiniusTower128.zero, "derivative X^1 = 0 (even index)")
    expectEqual(deriv[2], c3, "derivative X^2 = c3")

    // Derivative of constant = 0
    let constDeriv = PackedBinaryPoly128(constant: c0).derivative()
    expect(constDeriv.isZero, "derivative of constant is zero")

    // Derivative of X = 1
    let xPoly = PackedBinaryPoly128(coeffs: [.zero, .one])
    let xDeriv = xPoly.derivative()
    expectEqual(xDeriv.degree, 0, "derivative of X has degree 0")
    expectEqual(xDeriv[0], BiniusTower128.one, "derivative of X is 1")
}

private func testPoly128ScalarMul() {
    let poly = PackedBinaryPoly128(coeffs: [makeTower(1), makeTower(2), makeTower(3)])
    let scalar = makeTower(7)
    let scaled = poly.scaled(by: scalar)

    for i in 0..<poly.count {
        expectEqual(scaled[i], poly[i] * scalar, "scaled coeff \(i)")
    }

    // Scale by zero = zero
    let zeroScaled = poly.scaled(by: .zero)
    expect(zeroScaled.isZero, "scaling by zero gives zero poly")

    // Scale by one = identity
    let oneScaled = poly.scaled(by: .one)
    for i in 0..<poly.count {
        expectEqual(oneScaled[i], poly[i], "scaling by one preserves coeff \(i)")
    }
}

// MARK: - Multilinear128 Tests

private func testMultilinearEval128() {
    // 2-variable multilinear: f(x0, x1) defined by evals on {0,1}^2
    // f(0,0)=a, f(1,0)=b, f(0,1)=c, f(1,1)=d
    let a = makeTower(10)
    let b = makeTower(20)
    let c = makeTower(30)
    let d = makeTower(40)
    let evals = [a, b, c, d]

    // f(0, 0) should be a
    let p00 = [BiniusTower128.zero, BiniusTower128.zero]
    let v00 = BiniusMultilinearPoly128.evaluate(evals: evals, at: p00)
    expectEqual(v00, a, "MLE(0,0) = a")

    // f(1, 0) should be b
    let p10 = [BiniusTower128.one, BiniusTower128.zero]
    let v10 = BiniusMultilinearPoly128.evaluate(evals: evals, at: p10)
    expectEqual(v10, b, "MLE(1,0) = b")

    // f(0, 1) should be c
    let p01 = [BiniusTower128.zero, BiniusTower128.one]
    let v01 = BiniusMultilinearPoly128.evaluate(evals: evals, at: p01)
    expectEqual(v01, c, "MLE(0,1) = c")

    // f(1, 1) should be d
    let p11 = [BiniusTower128.one, BiniusTower128.one]
    let v11 = BiniusMultilinearPoly128.evaluate(evals: evals, at: p11)
    expectEqual(v11, d, "MLE(1,1) = d")

    // 1-variable case
    let evals1 = [makeTower(5), makeTower(9)]
    let r = makeTower(42)
    let eval1 = BiniusMultilinearPoly128.evaluate(evals: evals1, at: [r])
    // f(r) = f(0) + r*(f(1)+f(0)) = evals1[0] + r*(evals1[0]+evals1[1])
    let expected1 = evals1[0] + (r * (evals1[0] + evals1[1]))
    expectEqual(eval1, expected1, "1-var multilinear eval")
}

private func testMultilinearPartialEval() {
    // Start with 2-variable multilinear, fix first variable to challenge r
    let evals = makeTowerVec(4, seed: 100)
    let r = makeTower(77)

    let folded = BiniusMultilinearPoly128.partialEval(evals: evals, challenge: r)
    expectEqual(folded.count, 2, "partial eval halves the table")

    // folded[j] = evals[2j] + r*(evals[2j] + evals[2j+1])
    for j in 0..<2 {
        let expected = evals[2*j] + (r * (evals[2*j] + evals[2*j+1]))
        expectEqual(folded[j], expected, "partial eval entry \(j)")
    }

    // Consistency: evaluating the folded table at the second challenge
    // should equal evaluating the full table at (r, s)
    let s = makeTower(88)
    let fullEval = BiniusMultilinearPoly128.evaluate(evals: evals, at: [r, s])
    let partialResult = BiniusMultilinearPoly128.evaluate(evals: folded, at: [s])
    expectEqual(fullEval, partialResult, "partial eval then eval = full eval")
}

private func testMultilinearSumcheckRound() {
    let evals = makeTowerVec(8, seed: 200)  // 3 variables

    let (s0, s1) = BiniusMultilinearPoly128.sumcheckRound(evals: evals)

    // s0 = sum of even-indexed elements
    var expectedS0 = BiniusTower128.zero
    for j in stride(from: 0, to: evals.count, by: 2) {
        expectedS0 = expectedS0 + evals[j]
    }
    expectEqual(s0, expectedS0, "sumcheck round s0 = sum of even")

    // s1 = sum of odd-indexed elements
    var expectedS1 = BiniusTower128.zero
    for j in stride(from: 1, to: evals.count, by: 2) {
        expectedS1 = expectedS1 + evals[j]
    }
    expectEqual(s1, expectedS1, "sumcheck round s1 = sum of odd")

    // s0 + s1 = sum of all (the claim)
    let totalSum = BiniusMultilinearPoly128.sumAll(evals: evals)
    expectEqual(s0 + s1, totalSum, "s0 + s1 = total sum")
}

private func testMultilinearSumAll() {
    let evals = makeTowerVec(4, seed: 300)
    let sum = BiniusMultilinearPoly128.sumAll(evals: evals)

    var expected = BiniusTower128.zero
    for e in evals { expected = expected + e }
    expectEqual(sum, expected, "sumAll = XOR of all evals")

    // Sum of all zeros = zero
    let zeros = [BiniusTower128](repeating: .zero, count: 4)
    expectEqual(BiniusMultilinearPoly128.sumAll(evals: zeros), .zero, "sum of zeros is zero")

    // Sum of element with itself = 0 (char 2)
    let elem = makeTower(42)
    let pair = [elem, elem]
    expectEqual(BiniusMultilinearPoly128.sumAll(evals: pair), .zero, "a + a = 0")
}

private func testMultilinearTensorProduct() {
    let a = [makeTower(1), makeTower(2)]
    let b = [makeTower(3), makeTower(4)]

    let tensor = BiniusMultilinearPoly128.tensorProduct(a, b)
    expectEqual(tensor.count, 4, "tensor product size = |a| * |b|")

    // tensor[i*|b| + j] = a[i] * b[j]
    expectEqual(tensor[0], a[0] * b[0], "tensor [0,0]")
    expectEqual(tensor[1], a[0] * b[1], "tensor [0,1]")
    expectEqual(tensor[2], a[1] * b[0], "tensor [1,0]")
    expectEqual(tensor[3], a[1] * b[1], "tensor [1,1]")
}

// MARK: - Additive FFT128 Tests

private func testAdditiveFFT128Roundtrip() {
    // Use primitive basis for small dimension (ensures linear independence)
    let k = 3
    let n = 1 << k
    let basis = BiniusAdditiveFFT128.primitiveBasis(dimension: k)

    let coeffs = makeTowerVec(n, seed: 500)

    // Forward then inverse should recover original coefficients
    let evals = BiniusAdditiveFFT128.forward(coeffs, basis: basis)
    let recovered = BiniusAdditiveFFT128.inverse(evals, basis: basis)

    for i in 0..<n {
        expectEqual(recovered[i], coeffs[i], "FFT roundtrip coeff \(i)")
    }

    // Inverse then forward should also roundtrip
    let evals2 = makeTowerVec(n, seed: 600)
    let interp = BiniusAdditiveFFT128.inverse(evals2, basis: basis)
    let reEval = BiniusAdditiveFFT128.forward(interp, basis: basis)

    for i in 0..<n {
        expectEqual(reEval[i], evals2[i], "IFFT-FFT roundtrip eval \(i)")
    }
}

private func testAdditiveFFT128Linearity() {
    // Forward FFT should be linear: FFT(a + b) = FFT(a) + FFT(b)
    let k = 2
    let n = 1 << k
    let basis = BiniusAdditiveFFT128.primitiveBasis(dimension: k)

    let a = makeTowerVec(n, seed: 700)
    let b = makeTowerVec(n, seed: 800)

    // a + b
    var aPlusB = [BiniusTower128](repeating: .zero, count: n)
    for i in 0..<n { aPlusB[i] = a[i] + b[i] }

    let fftA = BiniusAdditiveFFT128.forward(a, basis: basis)
    let fftB = BiniusAdditiveFFT128.forward(b, basis: basis)
    let fftSum = BiniusAdditiveFFT128.forward(aPlusB, basis: basis)

    for i in 0..<n {
        expectEqual(fftSum[i], fftA[i] + fftB[i], "FFT linearity at index \(i)")
    }
}

private func testCanonicalBasis() {
    // Check that canonical basis elements are nonzero and distinct
    let k = 4
    let basis = BiniusAdditiveFFT128.canonicalBasis(dimension: k)
    expectEqual(basis.count, k, "canonical basis has k elements")

    for i in 0..<k {
        expect(!basis[i].isZero, "basis element \(i) is nonzero")
    }

    // Check distinctness
    for i in 0..<k {
        for j in (i+1)..<k {
            expect(basis[i] != basis[j], "basis elements \(i) and \(j) are distinct")
        }
    }
}

private func testPrimitiveBasis() {
    // Check that primitive basis elements are nonzero, distinct
    let k = 5
    let basis = BiniusAdditiveFFT128.primitiveBasis(dimension: k)
    expectEqual(basis.count, k, "primitive basis has k elements")

    for i in 0..<k {
        expect(!basis[i].isZero, "primitive basis element \(i) is nonzero")
    }

    // Check linear independence: all 2^k XOR combinations are distinct
    let n = 1 << k
    var seen = Set<String>()
    for mask in 0..<n {
        var elem = BiniusTower128.zero
        for i in 0..<k {
            if (mask >> i) & 1 == 1 {
                elem = elem + basis[i]
            }
        }
        let key = "\(elem.w0)_\(elem.w1)_\(elem.w2)_\(elem.w3)"
        seen.insert(key)
    }
    expectEqual(seen.count, n, "primitive basis spans 2^k distinct elements")
}

// MARK: - Reed-Solomon128 Tests

private func testReedSolomonRoundtrip() {
    // Encode a message and decode back
    let k = 2  // message size = 2^2 = 4
    let logRate = 1  // rate = 1/2, codeword size = 8
    let logN = k + logRate  // = 3
    let basis = BiniusAdditiveFFT128.primitiveBasis(dimension: logN)

    let message = makeTowerVec(1 << k, seed: 900)

    let codeword = BiniusReedSolomon128.encode(message: message, logRate: logRate, basis: basis)
    expectEqual(codeword.count, 1 << logN, "codeword has correct length")

    let decoded = BiniusReedSolomon128.decode(codeword: codeword, messageLength: 1 << k, basis: basis)
    expectEqual(decoded.count, 1 << k, "decoded has message length")

    for i in 0..<(1 << k) {
        expectEqual(decoded[i], message[i], "decoded message coeff \(i)")
    }
}

private func testReedSolomonValidCodeword() {
    let k = 2
    let logRate = 1
    let logN = k + logRate
    let basis = BiniusAdditiveFFT128.primitiveBasis(dimension: logN)
    let message = makeTowerVec(1 << k, seed: 1000)

    let codeword = BiniusReedSolomon128.encode(message: message, logRate: logRate, basis: basis)

    // A properly encoded codeword should be valid
    let isValid = BiniusReedSolomon128.isValidCodeword(codeword, messageLength: 1 << k, basis: basis)
    expect(isValid, "encoded codeword is valid")

    // Corrupt one position
    var corrupted = codeword
    corrupted[0] = corrupted[0] + BiniusTower128.one
    let isCorruptedValid = BiniusReedSolomon128.isValidCodeword(corrupted, messageLength: 1 << k, basis: basis)
    expect(!isCorruptedValid, "corrupted codeword is not valid")
}

private func testReedSolomonFRIFold() {
    // FRI folding: c'[j] = c[2j] + alpha * c[2j+1]
    let codeword = makeTowerVec(8, seed: 1100)
    let alpha = makeTower(42)

    let folded = BiniusReedSolomon128.foldCodeword(codeword, challenge: alpha)
    expectEqual(folded.count, 4, "folded codeword has half length")

    for j in 0..<4 {
        let expected = codeword[2*j] + (alpha * codeword[2*j+1])
        expectEqual(folded[j], expected, "fold entry \(j)")
    }

    // Folding with alpha=0 should give even-indexed elements
    let foldedZero = BiniusReedSolomon128.foldCodeword(codeword, challenge: .zero)
    for j in 0..<4 {
        expectEqual(foldedZero[j], codeword[2*j], "fold with alpha=0 gives even elements")
    }
}

private func testReedSolomonProximity() {
    let k = 2
    let logRate = 1
    let logN = k + logRate
    let basis = BiniusAdditiveFFT128.primitiveBasis(dimension: logN)
    let message = makeTowerVec(1 << k, seed: 1200)

    let codeword = BiniusReedSolomon128.encode(message: message, logRate: logRate, basis: basis)

    // Perfect codeword has distance 0
    let dist0 = BiniusReedSolomon128.proximityDistance(codeword, messageLength: 1 << k, basis: basis)
    expectEqual(dist0, 0, "valid codeword has distance 0")

    // Corrupt one position
    var corrupted = codeword
    corrupted[3] = corrupted[3] + makeTower(99)
    let dist1 = BiniusReedSolomon128.proximityDistance(corrupted, messageLength: 1 << k, basis: basis)
    expect(dist1 > 0, "corrupted codeword has positive distance")
}

// MARK: - Sumcheck128 Tests

private func testSumcheckProverState() {
    let evals = makeTowerVec(4, seed: 1300)  // 2 variables
    let state = BiniusSumcheck128.ProverState(evals: evals)

    expectEqual(state.numVars, 2, "2-variable sumcheck")
    expect(!state.isComplete, "not complete at start")

    // Current sum should be XOR of all evals
    var expectedSum = BiniusTower128.zero
    for e in evals { expectedSum = expectedSum + e }
    expectEqual(state.currentSum, expectedSum, "initial sum matches")
}

private func testSumcheckVerifyRound() {
    let evals = makeTowerVec(4, seed: 1400)
    let state = BiniusSumcheck128.ProverState(evals: evals)

    let (s0, s1) = state.roundPoly()
    let claimedSum = state.currentSum

    // Verify: s0 + s1 should equal claimed sum
    expectEqual(s0 + s1, claimedSum, "round poly sums to claimed value")

    // verifyRound should return valid=true
    let r = makeTower(55)
    let (valid, newSum) = BiniusSumcheck128.verifyRound(
        s0: s0, s1: s1, claimedSum: claimedSum, challenge: r)
    expect(valid, "verifyRound returns valid for correct sum")

    // New sum should equal S(r) = s0 + r*(s0+s1)
    let expectedNew = s0 + (r * (s0 + s1))
    expectEqual(newSum, expectedNew, "new sum matches S(r)")

    // Verify with wrong claimed sum should fail
    let wrongClaim = claimedSum + BiniusTower128.one
    let (validWrong, _) = BiniusSumcheck128.verifyRound(
        s0: s0, s1: s1, claimedSum: wrongClaim, challenge: r)
    expect(!validWrong, "verifyRound rejects wrong claimed sum")
}

private func testSumcheckFullProtocol() {
    let evals = makeTowerVec(8, seed: 1500)  // 3 variables
    let seed = makeTower(42)

    let (finalEval, point) = BiniusSumcheck128.runProtocol(evals: evals, challengeSeed: seed)
    expectEqual(point.count, 3, "protocol produces 3 challenges")

    // The final eval should equal MLE evaluated at the random point
    let mleEval = BiniusMultilinearPoly128.evaluate(evals: evals, at: point)
    expectEqual(finalEval, mleEval, "sumcheck final eval matches MLE")

    // Smaller case: 1 variable
    let evals1 = makeTowerVec(2, seed: 1600)
    let (finalEval1, point1) = BiniusSumcheck128.runProtocol(evals: evals1, challengeSeed: seed)
    expectEqual(point1.count, 1, "1-var protocol produces 1 challenge")
    let mle1 = BiniusMultilinearPoly128.evaluate(evals: evals1, at: point1)
    expectEqual(finalEval1, mle1, "1-var sumcheck final eval matches")
}

// MARK: - PackedGF2Poly Tests

private func testPackedGF2PolyBasics() {
    // Zero and one
    let z = PackedGF2Poly.zero
    let o = PackedGF2Poly.one
    expect(PackedGF2Poly.equal(z, (0, 0)), "zero is (0,0)")
    expect(PackedGF2Poly.equal(o, (1, 0)), "one is (1,0)")

    // Degree
    expectEqual(PackedGF2Poly.degree(z), -1, "degree of zero is -1")
    expectEqual(PackedGF2Poly.degree(o), 0, "degree of one is 0")
    expectEqual(PackedGF2Poly.degree(PackedGF2Poly.x), 1, "degree of X is 1")

    // High degree
    let highPoly: PackedGF2Poly.Poly128 = (0, 1 << 63)
    expectEqual(PackedGF2Poly.degree(highPoly), 127, "degree of X^127 is 127")

    // Coefficient access
    let poly: PackedGF2Poly.Poly128 = (0b1011, 0)  // 1 + X + X^3
    expectEqual(PackedGF2Poly.coeff(poly, at: 0), 1, "coeff 0")
    expectEqual(PackedGF2Poly.coeff(poly, at: 1), 1, "coeff 1")
    expectEqual(PackedGF2Poly.coeff(poly, at: 2), 0, "coeff 2")
    expectEqual(PackedGF2Poly.coeff(poly, at: 3), 1, "coeff 3")

    // Set coefficient
    var p = PackedGF2Poly.zero
    PackedGF2Poly.setCoeff(&p, at: 5, to: 1)
    expectEqual(PackedGF2Poly.coeff(p, at: 5), 1, "set coeff 5")
    expectEqual(PackedGF2Poly.degree(p), 5, "degree after set")
    PackedGF2Poly.setCoeff(&p, at: 5, to: 0)
    expectEqual(PackedGF2Poly.coeff(p, at: 5), 0, "clear coeff 5")
}

private func testPackedGF2PolyMul() {
    // (1 + X) * (1 + X) = 1 + X^2 in GF(2)[X]
    let onePlusX: PackedGF2Poly.Poly128 = (3, 0)  // 0b11
    let (prodLo, prodHi) = PackedGF2Poly.mul(onePlusX, onePlusX)
    // Expected: 1 + X^2 = 0b101 = 5
    expectEqual(prodLo.lo, 5, "(1+X)^2 lo.lo = 5")
    expectEqual(prodLo.hi, 0, "(1+X)^2 lo.hi = 0")
    expect(PackedGF2Poly.equal(prodHi, PackedGF2Poly.zero), "(1+X)^2 hi = 0")

    // (1 + X) * (1 + X + X^2) = 1 + X^3 in GF(2)[X]
    // (1+X)=0b11, (1+X+X^2)=0b111
    let triPoly: PackedGF2Poly.Poly128 = (7, 0)
    let (prod2Lo, prod2Hi) = PackedGF2Poly.mul(onePlusX, triPoly)
    // 1 + X^3 = 0b1001 = 9
    expectEqual(prod2Lo.lo, 9, "(1+X)*(1+X+X^2) lo.lo = 9")

    // Multiply by one = identity
    let a: PackedGF2Poly.Poly128 = (0xDEADBEEF, 0)
    let (aTimesOneLo, aTimesOneHi) = PackedGF2Poly.mul(a, PackedGF2Poly.one)
    expectEqual(aTimesOneLo.lo, a.lo, "a*1 lo = a lo")
    expectEqual(aTimesOneLo.hi, a.hi, "a*1 hi = a hi")

    // Multiply by zero = zero
    let (aTimesZeroLo, _) = PackedGF2Poly.mul(a, PackedGF2Poly.zero)
    expect(PackedGF2Poly.equal(aTimesZeroLo, PackedGF2Poly.zero), "a*0 lo = 0")
}

private func testPackedGF2PolyDivMod() {
    // Divide X^3 + 1 by X + 1 in GF(2)[X]
    // X^3 + 1 = (X + 1)(X^2 + X + 1) + 0
    let dividend: PackedGF2Poly.Poly128 = (0b1001, 0)  // X^3 + 1
    let divisor: PackedGF2Poly.Poly128 = (0b11, 0)     // X + 1

    let (q, r) = PackedGF2Poly.divmod(dividend, by: divisor)
    // q = X^2 + X + 1 = 0b111 = 7
    expectEqual(q.lo, 7, "quotient of (X^3+1)/(X+1) = X^2+X+1")
    // r = 0
    expect(PackedGF2Poly.equal(r, PackedGF2Poly.zero), "remainder is zero")

    // Verify: q * divisor + r = dividend
    let (checkLo, _) = PackedGF2Poly.mul(q, divisor)
    let reconstructed = PackedGF2Poly.add(checkLo, r)
    expectEqual(reconstructed.lo, dividend.lo, "divmod reconstruction lo")
    expectEqual(reconstructed.hi, dividend.hi, "divmod reconstruction hi")

    // Divide lower degree by higher degree: q=0, r=self
    let (q2, r2) = PackedGF2Poly.divmod(PackedGF2Poly.one, by: divisor)
    expect(PackedGF2Poly.equal(q2, PackedGF2Poly.zero), "1/(X+1) quotient is 0")
    expect(PackedGF2Poly.equal(r2, PackedGF2Poly.one), "1/(X+1) remainder is 1")
}

private func testPackedGF2PolyGCD() {
    // gcd(X^3 + 1, X + 1) = X + 1 (since X^3+1 = (X+1)(X^2+X+1))
    let a: PackedGF2Poly.Poly128 = (0b1001, 0)  // X^3 + 1
    let b: PackedGF2Poly.Poly128 = (0b11, 0)    // X + 1
    let g = PackedGF2Poly.gcd(a, b)
    // GCD should be X+1 (up to scalar, but in GF(2) the leading coeff is always 1)
    expectEqual(PackedGF2Poly.degree(g), 1, "gcd degree is 1")

    // gcd(X^2+1, X+1) = X+1 (since X^2+1 = (X+1)^2 in GF(2))
    let c: PackedGF2Poly.Poly128 = (0b101, 0)  // X^2 + 1
    let g2 = PackedGF2Poly.gcd(c, b)
    expectEqual(PackedGF2Poly.degree(g2), 1, "gcd(X^2+1, X+1) degree 1")

    // gcd of coprime polynomials = 1
    // X^2 + X + 1 and X + 1: (X+1) does not divide X^2+X+1 (eval at 1: 1+1+1=1 != 0)
    let d: PackedGF2Poly.Poly128 = (0b111, 0)  // X^2 + X + 1
    let g3 = PackedGF2Poly.gcd(d, b)
    expectEqual(PackedGF2Poly.degree(g3), 0, "coprime gcd has degree 0")
}

private func testPackedGF2PolyEvalAtBit() {
    // P(X) = 1 + X + X^3 = 0b1011
    let p: PackedGF2Poly.Poly128 = (0b1011, 0)

    // P(0) = constant = 1
    expectEqual(PackedGF2Poly.evalAtBit(p, bit: 0), 1, "P(0) = 1")

    // P(1) = 1 + 1 + 1 = 1 (parity of coefficients, 3 ones)
    expectEqual(PackedGF2Poly.evalAtBit(p, bit: 1), 1, "P(1) = 1")

    // X^2 + X = 0b110: P(0)=0, P(1) = parity of 2 ones = 0
    let q: PackedGF2Poly.Poly128 = (0b110, 0)
    expectEqual(PackedGF2Poly.evalAtBit(q, bit: 0), 0, "Q(0) = 0")
    expectEqual(PackedGF2Poly.evalAtBit(q, bit: 1), 0, "Q(1) = 0")

    // Zero poly
    expectEqual(PackedGF2Poly.evalAtBit(PackedGF2Poly.zero, bit: 0), 0, "zero(0) = 0")
    expectEqual(PackedGF2Poly.evalAtBit(PackedGF2Poly.zero, bit: 1), 0, "zero(1) = 0")
}

// MARK: - Engine Tests

private func testEngineBatchMultilinearEval() {
    let engine = GPUBiniusPolyEngine.shared

    // Two polynomials, same evaluation point
    let evals1 = makeTowerVec(4, seed: 2000)
    let evals2 = makeTowerVec(4, seed: 2100)
    let point = [makeTower(33), makeTower(44)]

    let results = engine.batchMultilinearEval(polys: [evals1, evals2], at: point)
    expectEqual(results.count, 2, "batch eval returns 2 results")

    // Each result should match individual evaluation
    let r1 = BiniusMultilinearPoly128.evaluate(evals: evals1, at: point)
    let r2 = BiniusMultilinearPoly128.evaluate(evals: evals2, at: point)
    expectEqual(results[0], r1, "batch eval poly 0 matches individual")
    expectEqual(results[1], r2, "batch eval poly 1 matches individual")

    // Empty input
    let empty = engine.batchMultilinearEval(polys: [], at: point)
    expectEqual(empty.count, 0, "empty batch returns empty")
}

private func testEngineBatchEval() {
    let engine = GPUBiniusPolyEngine.shared

    let poly = PackedBinaryPoly128(coeffs: [makeTower(1), makeTower(2), makeTower(3)])
    let points = [makeTower(10), makeTower(20), makeTower(30), .zero, .one]

    let results = engine.batchEval(poly: poly, at: points)
    expectEqual(results.count, 5, "batch eval at 5 points")

    for i in 0..<points.count {
        let expected = poly.evaluate(at: points[i])
        expectEqual(results[i], expected, "batch eval point \(i)")
    }
}

private func testEngineInterpolation() {
    let engine = GPUBiniusPolyEngine.shared

    // Interpolate from 3 points
    let x0 = makeTower(10)
    let x1 = makeTower(20)
    let x2 = makeTower(30)
    let y0 = makeTower(100)
    let y1 = makeTower(200)
    let y2 = makeTower(300)

    let points: [(x: BiniusTower128, y: BiniusTower128)] = [
        (x0, y0), (x1, y1), (x2, y2)
    ]

    let poly = engine.interpolate(points: points)

    // The interpolated polynomial should pass through all points
    for (i, pt) in points.enumerated() {
        let eval = poly.evaluate(at: pt.x)
        expectEqual(eval, pt.y, "interpolated poly passes through point \(i)")
    }

    // Degree should be at most n-1
    expect(poly.normalized().degree <= 2, "interpolated poly degree <= n-1")

    // Single point interpolation
    let singlePoly = engine.interpolate(points: [(x0, y0)])
    expectEqual(singlePoly.evaluate(at: x0), y0, "single point interpolation")
    expectEqual(singlePoly.degree, 0, "single point gives constant poly")

    // Empty interpolation
    let emptyPoly = engine.interpolate(points: [])
    expect(emptyPoly.isZero, "empty interpolation gives zero poly")
}

private func testEngineFRIFoldPoly() {
    let engine = GPUBiniusPolyEngine.shared

    // P(X) = c0 + c1*X + c2*X^2 + c3*X^3
    // P_even(Y) = c0 + c2*Y, P_odd(Y) = c1 + c3*Y
    // Folded: P'(Y) = P_even(Y) + alpha*P_odd(Y)
    //       = (c0 + alpha*c1) + (c2 + alpha*c3)*Y
    let c0 = makeTower(1)
    let c1 = makeTower(2)
    let c2 = makeTower(3)
    let c3 = makeTower(4)
    let poly = PackedBinaryPoly128(coeffs: [c0, c1, c2, c3])
    let alpha = makeTower(7)

    let folded = engine.friFoldPoly(poly, challenge: alpha)

    let expectedConst = c0 + (alpha * c1)
    let expectedLinear = c2 + (alpha * c3)
    expectEqual(folded[0], expectedConst, "FRI fold constant coeff")
    expectEqual(folded[1], expectedLinear, "FRI fold linear coeff")
    expect(folded.degree <= 1, "FRI fold halves degree")

    // Fold with alpha=0 gives even part
    let foldedZero = engine.friFoldPoly(poly, challenge: .zero)
    expectEqual(foldedZero[0], c0, "fold alpha=0 constant = c0")
    expectEqual(foldedZero[1], c2, "fold alpha=0 linear = c2")
}

private func testEngineLDE() {
    let engine = GPUBiniusPolyEngine.shared

    let smallK = 2
    let logBlowup = 1
    let largeK = smallK + logBlowup
    let smallBasis = BiniusAdditiveFFT128.primitiveBasis(dimension: smallK)
    let largeBasis = BiniusAdditiveFFT128.primitiveBasis(dimension: largeK)

    let evals = makeTowerVec(1 << smallK, seed: 3000)

    let extended = engine.lowDegreeExtend(evals: evals, smallBasis: smallBasis, largeBasis: largeBasis)
    expectEqual(extended.count, 1 << largeK, "LDE has correct length")

    // The first 2^smallK evaluations of the inverse FFT should match
    // the original coefficients
    let extCoeffs = BiniusAdditiveFFT128.inverse(extended, basis: largeBasis)
    let origCoeffs = BiniusAdditiveFFT128.inverse(evals, basis: smallBasis)

    // The low-degree coefficients should match
    for i in 0..<origCoeffs.count {
        expectEqual(extCoeffs[i], origCoeffs[i], "LDE low coeff \(i) matches")
    }

    // High-degree coefficients should be zero
    for i in origCoeffs.count..<extCoeffs.count {
        expectEqual(extCoeffs[i], BiniusTower128.zero, "LDE high coeff \(i) is zero")
    }
}

private func testEngineVanishingPoly() {
    let engine = GPUBiniusPolyEngine.shared

    let points = [makeTower(5), makeTower(10), makeTower(15)]
    let vp = engine.vanishingPoly(points: points)

    expectEqual(vp.degree, 3, "vanishing poly for 3 points has degree 3")

    // Should vanish at all given points
    for (i, pt) in points.enumerated() {
        let val = vp.evaluate(at: pt)
        expectEqual(val, BiniusTower128.zero, "vanishing poly zero at point \(i)")
    }

    // Should NOT vanish at a random other point (with high probability)
    let other = makeTower(999)
    let otherVal = vp.evaluate(at: other)
    // This is probabilistic but extremely unlikely to be zero for random input
    expect(!otherVal.isZero, "vanishing poly nonzero at random point")

    // Empty points: constant 1
    let emptyVP = engine.vanishingPoly(points: [])
    expectEqual(emptyVP.degree, 0, "empty vanishing poly degree 0")
    expectEqual(emptyVP[0], BiniusTower128.one, "empty vanishing poly = 1")
}

private func testEngineDegreeBound() {
    let engine = GPUBiniusPolyEngine.shared

    let k = 3
    let n = 1 << k
    let basis = BiniusAdditiveFFT128.primitiveBasis(dimension: k)

    // Create a low-degree polynomial (degree < 4) and evaluate via FFT
    var coeffs = [BiniusTower128](repeating: .zero, count: n)
    coeffs[0] = makeTower(1)
    coeffs[1] = makeTower(2)
    coeffs[2] = makeTower(3)
    coeffs[3] = makeTower(4)
    // coeffs[4..7] are zero -> degree < 4

    let evals = BiniusAdditiveFFT128.forward(coeffs, basis: basis)

    // Should pass degree bound 4
    expect(engine.checkDegreeBound(evals: evals, bound: 4, basis: basis),
           "degree < 4 passes bound 4")

    // Should pass degree bound 8
    expect(engine.checkDegreeBound(evals: evals, bound: 8, basis: basis),
           "degree < 4 passes bound 8")

    // Should fail degree bound 2
    expect(!engine.checkDegreeBound(evals: evals, bound: 2, basis: basis),
           "degree 3 fails bound 2")
}

private func testEngineCompose() {
    let engine = GPUBiniusPolyEngine.shared

    // P(X) = a + b*X, Q(X) = c + d*X
    // P(Q(X)) = a + b*(c + d*X) = (a + b*c) + b*d*X
    let a = makeTower(1)
    let b = makeTower(2)
    let c = makeTower(3)
    let d = makeTower(4)

    let p = PackedBinaryPoly128(coeffs: [a, b])
    let q = PackedBinaryPoly128(coeffs: [c, d])

    let composed = engine.compose(p, with: q)

    // Verify by evaluating at a test point
    let x = makeTower(10)
    let qAtX = q.evaluate(at: x)
    let pAtQX = p.evaluate(at: qAtX)
    let composedAtX = composed.evaluate(at: x)
    expectEqual(composedAtX, pAtQX, "P(Q(x)) matches direct evaluation")

    // Compose with zero -> P(0) = constant
    let compZero = engine.compose(p, with: .zero)
    expectEqual(compZero.evaluate(at: x), a, "P(0) = constant term")
}

private func testEngineLinearCombination() {
    let engine = GPUBiniusPolyEngine.shared

    let n = 4
    let evals1 = makeTowerVec(n, seed: 4000)
    let evals2 = makeTowerVec(n, seed: 4100)
    let alpha = makeTower(7)
    let beta = makeTower(11)

    let result = engine.linearCombination(scalars: [alpha, beta], polys: [evals1, evals2])
    expectEqual(result.count, n, "linear combination has correct length")

    for i in 0..<n {
        let expected = (alpha * evals1[i]) + (beta * evals2[i])
        expectEqual(result[i], expected, "linear combination entry \(i)")
    }

    // Empty linear combination
    let emptyResult = engine.linearCombination(scalars: [], polys: [])
    expectEqual(emptyResult.count, 0, "empty linear combination")
}

private func testEngineBatchInnerProducts() {
    let engine = GPUBiniusPolyEngine.shared

    let a1 = makeTowerVec(4, seed: 5000)
    let b1 = makeTowerVec(4, seed: 5100)
    let a2 = makeTowerVec(4, seed: 5200)
    let b2 = makeTowerVec(4, seed: 5300)

    let results = engine.batchInnerProducts([(a1, b1), (a2, b2)])
    expectEqual(results.count, 2, "batch inner products returns 2 results")

    // Verify each inner product
    let ip1 = BiniusTowerBatch.innerProduct(a1, b1)
    let ip2 = BiniusTowerBatch.innerProduct(a2, b2)
    expectEqual(results[0], ip1, "batch inner product 0")
    expectEqual(results[1], ip2, "batch inner product 1")
}

private func testEngineBatchFFT() {
    let engine = GPUBiniusPolyEngine.shared

    let k = 2
    let n = 1 << k
    let basis = BiniusAdditiveFFT128.primitiveBasis(dimension: k)

    let poly1 = makeTowerVec(n, seed: 6000)
    let poly2 = makeTowerVec(n, seed: 6100)

    // Batch forward FFT
    let evalSets = engine.batchForwardFFT(polys: [poly1, poly2], basis: basis)
    expectEqual(evalSets.count, 2, "batch FFT produces 2 eval sets")

    // Each should match individual FFT
    let fft1 = BiniusAdditiveFFT128.forward(poly1, basis: basis)
    let fft2 = BiniusAdditiveFFT128.forward(poly2, basis: basis)

    for i in 0..<n {
        expectEqual(evalSets[0][i], fft1[i], "batch FFT poly 0 eval \(i)")
        expectEqual(evalSets[1][i], fft2[i], "batch FFT poly 1 eval \(i)")
    }

    // Batch inverse FFT roundtrip
    let recovered = engine.batchInverseFFT(evalSets: evalSets, basis: basis)
    for i in 0..<n {
        expectEqual(recovered[0][i], poly1[i], "batch IFFT poly 0 coeff \(i)")
        expectEqual(recovered[1][i], poly2[i], "batch IFFT poly 1 coeff \(i)")
    }
}
