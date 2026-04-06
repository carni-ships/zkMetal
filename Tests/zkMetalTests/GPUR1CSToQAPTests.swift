// GPU R1CS-to-QAP Engine tests
import zkMetal
import Foundation

private func frEqual(_ a: Fr, _ b: Fr) -> Bool {
    return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
           a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

public func runGPUR1CSToQAPTests() {
    testSimpleQuotient()
    testAdditionCircuitQAP()
    testMultiConstraintQAP()
    testMarlinConversion()
    testCosetEvaluation()
    testInverseCosetEval()
    testVanishingPolynomial()
    testDomainSizeComputation()
    testDomainGeneration()
    testDensityAnalysis()
    testBatchSparseMatVec()
    testQuotientSatisfiedCircuit()
    testQuotientUnsatisfiedCircuit()
    testCosetRoundTrip()
}

// MARK: - Test 1: Simple multiplication quotient

private func testSimpleQuotient() {
    suite("R1CS-to-QAP: simple multiplication quotient")

    // Circuit: a * b = c
    // Wires: [1, a, b, c]
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    bE.append(R1CSEntry(row: 0, col: 2, val: one))
    cE.append(R1CSEntry(row: 0, col: 3, val: one))

    let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 2,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    let a = frFromInt(3)
    let b = frFromInt(7)
    let c = frMul(a, b)  // 21
    let witness = [Fr.one, a, b, c]

    let engine = GPUR1CSToQAPEngine()

    do {
        let result = try engine.convert(r1cs: r1cs, witness: witness)

        expect(result.domainSize >= 1, "domain size >= numConstraints")
        expect(result.logDomainSize >= 0, "logDomainSize is valid")
        expect(result.hPoly.count == result.domainSize, "H poly has domainSize coefficients")

        // A*z should be [a] = [3], B*z = [b] = [7], C*z = [c] = [21]
        expect(result.aEvals.count == 1, "1 constraint => 1 A eval")
        expect(frEqual(result.aEvals[0], a), "A*z[0] = a")
        expect(frEqual(result.bEvals[0], b), "B*z[0] = b")
        expect(frEqual(result.cEvals[0], c), "C*z[0] = c")

        // For a satisfied circuit, the quotient should exist (no remainder)
        // The numerator A*B - C is zero on the evaluation domain, so H is well-defined
        // We verify by checking that H is not all-zero for a nontrivial circuit
        // Actually for 1 constraint with A*B=C, the numerator polynomial is zero everywhere
        // on the domain, so H could be all zeros. That is correct behavior.
    } catch {
        expect(false, "convert threw: \(error)")
    }
}

// MARK: - Test 2: Addition circuit QAP

private func testAdditionCircuitQAP() {
    suite("R1CS-to-QAP: addition circuit")

    // Circuit: (a + b) * 1 = c
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    aE.append(R1CSEntry(row: 0, col: 2, val: one))
    bE.append(R1CSEntry(row: 0, col: 0, val: one))
    cE.append(R1CSEntry(row: 0, col: 3, val: one))

    let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 2,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    let a = frFromInt(10)
    let b = frFromInt(20)
    let c = frAdd(a, b)  // 30
    let witness = [Fr.one, a, b, c]

    let engine = GPUR1CSToQAPEngine()

    do {
        let result = try engine.convert(r1cs: r1cs, witness: witness)

        expect(result.domainSize >= 1, "domain size valid")
        expect(result.aEvals.count == 1, "1 constraint")

        // A*z should be a + b = 30
        let expectedAz = frAdd(a, b)
        expect(frEqual(result.aEvals[0], expectedAz), "A*z = a + b")
        expect(frEqual(result.bEvals[0], Fr.one), "B*z = 1 (constant wire)")
        expect(frEqual(result.cEvals[0], c), "C*z = c")
    } catch {
        expect(false, "convert threw: \(error)")
    }
}

// MARK: - Test 3: Multi-constraint QAP

private func testMultiConstraintQAP() {
    suite("R1CS-to-QAP: multi-constraint")

    // Circuit: a*b=c, c*d=e
    // Wires: [1, a, b, d, c, e]
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    bE.append(R1CSEntry(row: 0, col: 2, val: one))
    cE.append(R1CSEntry(row: 0, col: 4, val: one))

    aE.append(R1CSEntry(row: 1, col: 4, val: one))
    bE.append(R1CSEntry(row: 1, col: 3, val: one))
    cE.append(R1CSEntry(row: 1, col: 5, val: one))

    let r1cs = R1CSInstance(numConstraints: 2, numVars: 6, numPublic: 3,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    let a = frFromInt(2)
    let b = frFromInt(5)
    let d = frFromInt(4)
    let c_val = frMul(a, b)        // 10
    let e_val = frMul(c_val, d)    // 40

    let witness = [Fr.one, a, b, d, c_val, e_val]

    let engine = GPUR1CSToQAPEngine()

    do {
        let result = try engine.convert(r1cs: r1cs, witness: witness)

        expect(result.domainSize >= 2, "domain size >= 2 constraints")
        expect(result.aEvals.count == 2, "2 A evals")
        expect(result.bEvals.count == 2, "2 B evals")
        expect(result.cEvals.count == 2, "2 C evals")

        // Verify evaluations
        expect(frEqual(result.aEvals[0], a), "A*z[0] = a")
        expect(frEqual(result.bEvals[0], b), "B*z[0] = b")
        expect(frEqual(result.cEvals[0], c_val), "C*z[0] = a*b")

        expect(frEqual(result.aEvals[1], c_val), "A*z[1] = c")
        expect(frEqual(result.bEvals[1], d), "B*z[1] = d")
        expect(frEqual(result.cEvals[1], e_val), "C*z[1] = c*d")

        // Quotient should have domainSize coefficients
        expect(result.hPoly.count == result.domainSize, "H poly length = domainSize")
    } catch {
        expect(false, "convert threw: \(error)")
    }
}

// MARK: - Test 4: Marlin conversion

private func testMarlinConversion() {
    suite("R1CS-to-QAP: Marlin format")

    // Circuit: a * b = c
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    bE.append(R1CSEntry(row: 0, col: 2, val: one))
    cE.append(R1CSEntry(row: 0, col: 3, val: one))

    let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 2,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    let a = frFromInt(6)
    let b = frFromInt(9)
    let c = frMul(a, b)  // 54
    let witness = [Fr.one, a, b, c]

    let engine = GPUR1CSToQAPEngine()

    do {
        let result = try engine.convertMarlin(r1cs: r1cs, witness: witness)

        expect(result.constraintDomainSize >= 1, "constraint domain size valid")
        expect(result.variableDomainSize >= 4, "variable domain size >= numVars")

        // z_A, z_B, z_C should have constraintDomainSize coefficients
        expect(result.zACoeffs.count == result.constraintDomainSize, "z_A length")
        expect(result.zBCoeffs.count == result.constraintDomainSize, "z_B length")
        expect(result.zCCoeffs.count == result.constraintDomainSize, "z_C length")

        // Quotient should also have constraintDomainSize coefficients
        expect(result.tCoeffs.count == result.constraintDomainSize, "t length")

        // The omega_H should be a root of unity
        let omegaN = frPow(result.omegaH, UInt64(result.constraintDomainSize))
        expect(frEqual(omegaN, Fr.one), "omega_H^|H| = 1")
    } catch {
        expect(false, "convertMarlin threw: \(error)")
    }
}

// MARK: - Test 5: Coset evaluation

private func testCosetEvaluation() {
    suite("R1CS-to-QAP: coset evaluation")

    let engine = GPUR1CSToQAPEngine()

    // Polynomial: f(x) = 1 + 2x + 3x^2 + 4x^3  (4 coefficients, logN=2)
    let coeffs = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let g = frFromInt(5)  // coset generator

    do {
        let result = try engine.evaluateOnCoset(coeffs: coeffs, cosetGen: g)

        expect(result.evaluations.count == 4, "4 evaluations")
        expect(result.domainSize == 4, "domain size 4")
        expect(frEqual(result.cosetGenerator, g), "coset gen preserved")

        // Verify by evaluating f(g * omega^i) directly for each coset point
        let logN = 2
        let omega = frRootOfUnity(logN: logN)
        var omPow = Fr.one

        for i in 0..<4 {
            let pt = frMul(g, omPow)

            // Horner: f(pt) = 1 + pt*(2 + pt*(3 + pt*4))
            var val = coeffs[3]
            val = frAdd(frMul(val, pt), coeffs[2])
            val = frAdd(frMul(val, pt), coeffs[1])
            val = frAdd(frMul(val, pt), coeffs[0])

            expect(frEqual(result.evaluations[i], val), "coset eval[\(i)] matches Horner")
            omPow = frMul(omPow, omega)
        }
    } catch {
        expect(false, "evaluateOnCoset threw: \(error)")
    }
}

// MARK: - Test 6: Inverse coset evaluation

private func testInverseCosetEval() {
    suite("R1CS-to-QAP: inverse coset eval")

    let engine = GPUR1CSToQAPEngine()

    // Start with known coefficients, coset-evaluate, then invert
    let coeffs = [frFromInt(7), frFromInt(11), frFromInt(13), frFromInt(17)]
    let g = frFromInt(3)

    do {
        let fwd = try engine.evaluateOnCoset(coeffs: coeffs, cosetGen: g)
        let recovered = try engine.inverseCosetEval(evals: fwd.evaluations, cosetGen: g)

        expect(recovered.count == 4, "recovered 4 coefficients")
        for i in 0..<4 {
            expect(frEqual(recovered[i], coeffs[i]), "recovered coeff[\(i)] matches original")
        }
    } catch {
        expect(false, "coset round-trip threw: \(error)")
    }
}

// MARK: - Test 7: Vanishing polynomial

private func testVanishingPolynomial() {
    suite("R1CS-to-QAP: vanishing polynomial")

    let engine = GPUR1CSToQAPEngine()

    // Z_H(omega^i) should be 0 for all i in domain
    let logN = 3
    let n = 1 << logN
    let omega = frRootOfUnity(logN: logN)
    var omPow = Fr.one

    for i in 0..<n {
        let zh = engine.evaluateVanishing(point: omPow, domainSize: n)
        expect(frEqual(zh, Fr.zero), "Z_H(omega^\(i)) = 0")
        omPow = frMul(omPow, omega)
    }

    // Z_H at a random point should be nonzero
    let randPt = frFromInt(42)
    let zhRand = engine.evaluateVanishing(point: randPt, domainSize: n)
    expect(!frEqual(zhRand, Fr.zero), "Z_H(42) != 0")

    // Z_H(x) = x^n - 1, so Z_H(2) = 2^8 - 1 = 255
    let pt2 = frFromInt(2)
    let zh2 = engine.evaluateVanishing(point: pt2, domainSize: n)
    let expected = frSub(frPow(pt2, UInt64(n)), Fr.one)
    expect(frEqual(zh2, expected), "Z_H(2) = 2^8 - 1 = 255")

    // Batch evaluation
    let points = [frFromInt(2), frFromInt(3), frFromInt(5)]
    let batch = engine.evaluateVanishingBatch(points: points, domainSize: n)
    expect(batch.count == 3, "3 batch results")
    for (i, pt) in points.enumerated() {
        let exp = frSub(frPow(pt, UInt64(n)), Fr.one)
        expect(frEqual(batch[i], exp), "batch Z_H[\(i)] correct")
    }
}

// MARK: - Test 8: Domain size computation

private func testDomainSizeComputation() {
    suite("R1CS-to-QAP: domain size computation")

    let (d1, l1) = GPUR1CSToQAPEngine.domainSizeFor(n: 1)
    expect(d1 == 1, "domainSize(1) = 1")
    expect(l1 == 0, "logN(1) = 0")

    let (d2, l2) = GPUR1CSToQAPEngine.domainSizeFor(n: 2)
    expect(d2 == 2, "domainSize(2) = 2")
    expect(l2 == 1, "logN(2) = 1")

    let (d5, l5) = GPUR1CSToQAPEngine.domainSizeFor(n: 5)
    expect(d5 == 8, "domainSize(5) = 8")
    expect(l5 == 3, "logN(5) = 3")

    let (d1024, l1024) = GPUR1CSToQAPEngine.domainSizeFor(n: 1024)
    expect(d1024 == 1024, "domainSize(1024) = 1024")
    expect(l1024 == 10, "logN(1024) = 10")

    let (d1025, l1025) = GPUR1CSToQAPEngine.domainSizeFor(n: 1025)
    expect(d1025 == 2048, "domainSize(1025) = 2048")
    expect(l1025 == 11, "logN(1025) = 11")
}

// MARK: - Test 9: Domain generation

private func testDomainGeneration() {
    suite("R1CS-to-QAP: domain generation")

    let engine = GPUR1CSToQAPEngine()
    let logN = 3
    let n = 1 << logN

    let domain = engine.generateDomain(logN: logN)
    expect(domain.count == n, "domain has 8 elements")
    expect(frEqual(domain[0], Fr.one), "domain[0] = 1")

    // omega^n should be 1 (wraparound)
    let omega = frRootOfUnity(logN: logN)
    let omegaN = frPow(omega, UInt64(n))
    expect(frEqual(omegaN, Fr.one), "omega^n = 1")

    // Each domain element should be omega^i
    var omPow = Fr.one
    for i in 0..<n {
        expect(frEqual(domain[i], omPow), "domain[\(i)] = omega^\(i)")
        omPow = frMul(omPow, omega)
    }

    // Coset domain
    let g = frFromInt(7)
    let cosetDomain = engine.generateCosetDomain(logN: logN, cosetGen: g)
    expect(cosetDomain.count == n, "coset domain has 8 elements")
    expect(frEqual(cosetDomain[0], g), "coset domain[0] = g")

    omPow = g
    for i in 0..<n {
        expect(frEqual(cosetDomain[i], omPow), "coset[\(i)] = g*omega^\(i)")
        omPow = frMul(omPow, omega)
    }
}

// MARK: - Test 10: Density analysis

private func testDensityAnalysis() {
    suite("R1CS-to-QAP: density analysis")

    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    // Constraint 0: w[1] * w[2] = w[4]  (3 entries)
    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    bE.append(R1CSEntry(row: 0, col: 2, val: one))
    cE.append(R1CSEntry(row: 0, col: 4, val: one))

    // Constraint 1: (w[1]+w[2]) * w[3] = w[5]  (4 entries: 2 in A)
    aE.append(R1CSEntry(row: 1, col: 1, val: one))
    aE.append(R1CSEntry(row: 1, col: 2, val: one))
    bE.append(R1CSEntry(row: 1, col: 3, val: one))
    cE.append(R1CSEntry(row: 1, col: 5, val: one))

    let r1cs = R1CSInstance(numConstraints: 2, numVars: 6, numPublic: 3,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    let engine = GPUR1CSToQAPEngine()
    let info = engine.analyzeDensity(r1cs: r1cs)

    expect(info.totalNonZero == 7, "7 total non-zero entries")
    expect(info.aNonZero == 3, "3 A entries")
    expect(info.bNonZero == 2, "2 B entries")
    expect(info.cNonZero == 2, "2 C entries")
    expect(info.avgEntriesPerRow == 3.5, "avg 3.5 per row")
    expect(info.maxEntriesPerRow == 4, "max 4 per row")
}

// MARK: - Test 11: Batch sparse mat-vec

private func testBatchSparseMatVec() {
    suite("R1CS-to-QAP: batch sparse mat-vec")

    let engine = GPUR1CSToQAPEngine()

    // Build a small circuit with known result
    let one = Fr.one
    let two = frFromInt(2)
    let three = frFromInt(3)

    // 3 entries: row 0 gets 1*w[0] + 2*w[1], row 1 gets 3*w[2]
    let entries = [
        R1CSEntry(row: 0, col: 0, val: one),
        R1CSEntry(row: 0, col: 1, val: two),
        R1CSEntry(row: 1, col: 2, val: three)
    ]

    let witness = [frFromInt(10), frFromInt(20), frFromInt(30)]

    // No batching
    let result1 = engine.sparseMatVecBatch(entries: entries, witness: witness, numRows: 2, batchSize: 0)
    expect(result1.count == 2, "2 rows")

    // row 0: 1*10 + 2*20 = 50
    let exp0 = frAdd(frMul(one, frFromInt(10)), frMul(two, frFromInt(20)))
    expect(frEqual(result1[0], exp0), "row 0 = 50")

    // row 1: 3*30 = 90
    let exp1 = frMul(three, frFromInt(30))
    expect(frEqual(result1[1], exp1), "row 1 = 90")

    // With small batch size (1 entry per batch)
    let result2 = engine.sparseMatVecBatch(entries: entries, witness: witness, numRows: 2, batchSize: 1)
    expect(frEqual(result2[0], exp0), "batched row 0 = 50")
    expect(frEqual(result2[1], exp1), "batched row 1 = 90")
}

// MARK: - Test 12: Quotient for satisfied circuit

private func testQuotientSatisfiedCircuit() {
    suite("R1CS-to-QAP: quotient satisfied circuit")

    // Build a 2-constraint circuit: a*b=c, c+1=d  (encoded as (c + 1*w[0]) * 1 = d)
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    // Constraint 0: w[1] * w[2] = w[3]
    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    bE.append(R1CSEntry(row: 0, col: 2, val: one))
    cE.append(R1CSEntry(row: 0, col: 3, val: one))

    // Constraint 1: (w[3] + w[0]) * w[0] = w[4]  =>  (c+1)*1 = d
    aE.append(R1CSEntry(row: 1, col: 3, val: one))
    aE.append(R1CSEntry(row: 1, col: 0, val: one))
    bE.append(R1CSEntry(row: 1, col: 0, val: one))
    cE.append(R1CSEntry(row: 1, col: 4, val: one))

    let r1cs = R1CSInstance(numConstraints: 2, numVars: 5, numPublic: 2,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    let a = frFromInt(3)
    let b = frFromInt(4)
    let c_val = frMul(a, b)  // 12
    let d_val = frAdd(c_val, Fr.one)  // 13
    let witness = [Fr.one, a, b, c_val, d_val]

    let engine = GPUR1CSToQAPEngine()

    do {
        let hPoly = try engine.computeQuotient(r1cs: r1cs, witness: witness)
        expect(hPoly.count >= 2, "H poly has at least domainSize coefficients")

        // Verify the R1CS is actually satisfied
        expect(r1cs.isSatisfied(z: witness), "R1CS is satisfied before QAP conversion")
    } catch {
        expect(false, "computeQuotient threw: \(error)")
    }
}

// MARK: - Test 13: Quotient for unsatisfied circuit

private func testQuotientUnsatisfiedCircuit() {
    suite("R1CS-to-QAP: quotient unsatisfied circuit")

    // Circuit: a * b = c (but we provide wrong c)
    let one = Fr.one
    var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

    aE.append(R1CSEntry(row: 0, col: 1, val: one))
    bE.append(R1CSEntry(row: 0, col: 2, val: one))
    cE.append(R1CSEntry(row: 0, col: 3, val: one))

    let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 2,
                            aEntries: aE, bEntries: bE, cEntries: cE)

    let a = frFromInt(3)
    let b = frFromInt(7)
    let wrongC = frFromInt(42)  // should be 21
    let witness = [Fr.one, a, b, wrongC]

    let engine = GPUR1CSToQAPEngine()

    // The circuit is NOT satisfied
    expect(!r1cs.isSatisfied(z: witness), "R1CS is NOT satisfied with wrong witness")

    // But computeQuotient still produces output (it does not check satisfaction)
    do {
        let hPoly = try engine.computeQuotient(r1cs: r1cs, witness: witness)
        expect(hPoly.count >= 1, "H poly produced even for unsatisfied circuit")
        // The H poly will NOT correctly divide the numerator, but the computation completes
    } catch {
        expect(false, "computeQuotient threw for unsatisfied circuit: \(error)")
    }
}

// MARK: - Test 14: Coset round-trip with larger polynomial

private func testCosetRoundTrip() {
    suite("R1CS-to-QAP: coset round-trip (8 coefficients)")

    let engine = GPUR1CSToQAPEngine()

    // 8-coefficient polynomial
    let coeffs: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
    let g = frFromInt(11)

    do {
        let fwd = try engine.evaluateOnCoset(coeffs: coeffs, cosetGen: g)
        expect(fwd.evaluations.count == 8, "8 evaluations")

        let recovered = try engine.inverseCosetEval(evals: fwd.evaluations, cosetGen: g)
        expect(recovered.count == 8, "8 recovered coefficients")

        for i in 0..<8 {
            expect(frEqual(recovered[i], coeffs[i]), "round-trip coeff[\(i)]")
        }
    } catch {
        expect(false, "coset round-trip threw: \(error)")
    }
}
