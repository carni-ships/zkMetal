// GPUMarlinPolyIOPTests — Comprehensive tests for GPU-accelerated Marlin polynomial IOP engine
//
// Tests: indexing, witness construction, quotient polynomial, sumcheck,
// holographic reduction, full prove+verify, GPU vs CPU equivalence,
// polynomial arithmetic, configuration modes, statistics, edge cases.

import Foundation
import zkMetal

public func runGPUMarlinPolyIOPTests() {
    suite("GPUMarlinPolyIOP")

    // Shared SRS setup
    let gen = bn254G1Generator()
    let secret: [UInt32] = [0xCAFE, 0xBEEF, 0xDEAD, 0x1234, 0x5678, 0x9ABC, 0xDEF0, 0x0001]
    let srsSecret = frFromLimbs(secret)
    let srs = KZGEngine.generateTestSRS(secret: secret, size: 512, generator: gen)

    // ================================================================
    // MARK: - Engine Initialization
    // ================================================================

    testEngineInit(srs: srs)
    testConvenienceInit(srs: srs)
    testConfigurationModes()

    // ================================================================
    // MARK: - Polynomial Arithmetic
    // ================================================================

    testPolyAdd(srs: srs)
    testPolySub(srs: srs)
    testPolyScalarMul(srs: srs)
    testPolyMul(srs: srs)
    testEvalVanishing(srs: srs)
    testCPUEvalPoly(srs: srs)
    testGPUEvalPoly(srs: srs)

    // ================================================================
    // MARK: - Poseidon2 Hashing
    // ================================================================

    testHashPair(srs: srs)
    testHashSequence(srs: srs)

    // ================================================================
    // MARK: - Indexing
    // ================================================================

    testIndexSimple(srs: srs, srsSecret: srsSecret)
    testIndexMultiConstraint(srs: srs, srsSecret: srsSecret)

    // ================================================================
    // MARK: - Witness Construction
    // ================================================================

    testWitnessPolynomials(srs: srs, srsSecret: srsSecret)

    // ================================================================
    // MARK: - Quotient Polynomial
    // ================================================================

    testQuotientPolynomial(srs: srs, srsSecret: srsSecret)

    // ================================================================
    // MARK: - Sumcheck
    // ================================================================

    testSumcheckPolys(srs: srs)
    testSumcheckConsistency(srs: srs)

    // ================================================================
    // MARK: - Holographic Reduction
    // ================================================================

    testHolographicReduction(srs: srs, srsSecret: srsSecret)

    // ================================================================
    // MARK: - Full Prove + Verify
    // ================================================================

    testProveVerifyMultiply(srs: srs, srsSecret: srsSecret)
    testProveVerify10Constraints(srs: srs, srsSecret: srsSecret)
    testWrongWitnessRejected(srs: srs, srsSecret: srsSecret)
    testIndexReuse(srs: srs, srsSecret: srsSecret)

    // ================================================================
    // MARK: - GPU vs CPU Equivalence
    // ================================================================

    testGPUvsCPUEquivalence(srs: srs, srsSecret: srsSecret)

    // ================================================================
    // MARK: - Statistics
    // ================================================================

    testStatisticsTracking(srs: srs, srsSecret: srsSecret)

    // ================================================================
    // MARK: - Verify Diagnostics
    // ================================================================

    testVerifyDiag(srs: srs, srsSecret: srsSecret)
}

// MARK: - Engine Initialization Tests

private func testEngineInit(srs: [PointAffine]) {
    suite("GPUMarlinPolyIOP — Engine Init")

    do {
        let kzg = try KZGEngine(srs: srs)
        let ntt = try NTTEngine()
        let engine = try GPUMarlinPolyIOPEngine(kzg: kzg, ntt: ntt)
        expect(engine.config.gpuEvalThreshold == 256, "Default eval threshold is 256")
        expect(engine.config.enableGPUPolyArith, "GPU poly arith enabled by default")
        expect(engine.config.enableGPUSigma, "GPU sigma enabled by default")
        expect(engine.lastStats.gpuEvalCount == 0, "Stats start at zero")
    } catch {
        expect(false, "Engine init threw: \(error)")
    }
}

private func testConvenienceInit(srs: [PointAffine]) {
    suite("GPUMarlinPolyIOP — Convenience Init")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)
        expect(engine.config.gpuEvalThreshold == 256, "Default config via convenience init")
    } catch {
        expect(false, "Convenience init threw: \(error)")
    }
}

private func testConfigurationModes() {
    suite("GPUMarlinPolyIOP — Configuration Modes")

    let defaultConfig = GPUMarlinPolyIOPConfig.default
    expect(defaultConfig.gpuEvalThreshold == 256, "Default eval threshold")
    expect(defaultConfig.enableGPUPolyArith, "Default: GPU poly arith on")
    expect(defaultConfig.enableGPUSigma, "Default: GPU sigma on")

    let forceGPU = GPUMarlinPolyIOPConfig.forceGPU
    expect(forceGPU.gpuEvalThreshold == 1, "ForceGPU: threshold=1")
    expect(forceGPU.gpuBatchInverseThreshold == 1, "ForceGPU: batch inverse threshold=1")

    let forceCPU = GPUMarlinPolyIOPConfig.forceCPU
    expect(forceCPU.gpuEvalThreshold == Int.max, "ForceCPU: threshold=max")
    expect(!forceCPU.enableGPUPolyArith, "ForceCPU: GPU poly arith off")
    expect(!forceCPU.enableGPUSigma, "ForceCPU: GPU sigma off")

    let custom = GPUMarlinPolyIOPConfig(
        gpuEvalThreshold: 64,
        gpuBatchInverseThreshold: 32,
        gpuNTTThreshold: 16,
        enableGPUPolyArith: false,
        enableGPUSigma: true
    )
    expectEqual(custom.gpuEvalThreshold, 64, "Custom eval threshold")
    expectEqual(custom.gpuBatchInverseThreshold, 32, "Custom batch inverse threshold")
    expectEqual(custom.gpuNTTThreshold, 16, "Custom NTT threshold")
    expect(!custom.enableGPUPolyArith, "Custom: poly arith off")
    expect(custom.enableGPUSigma, "Custom: sigma on")
}

// MARK: - Polynomial Arithmetic Tests

private func testPolyAdd(srs: [PointAffine]) {
    suite("GPUMarlinPolyIOP — Poly Add")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)

        // [1, 2, 3] + [4, 5] = [5, 7, 3]
        let a = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let b = [frFromInt(4), frFromInt(5)]
        let c = engine.polyAdd(a, b)
        expectEqual(c.count, 3, "Poly add result length")
        expectEqual(c[0], frFromInt(5), "Poly add coeff 0")
        expectEqual(c[1], frFromInt(7), "Poly add coeff 1")
        expectEqual(c[2], frFromInt(3), "Poly add coeff 2")

        // Empty + non-empty
        let d = engine.polyAdd([], [frFromInt(42)])
        expectEqual(d.count, 1, "Empty + non-empty length")
        expectEqual(d[0], frFromInt(42), "Empty + non-empty value")

        // Empty + empty
        let e = engine.polyAdd([], [])
        expectEqual(e.count, 0, "Empty + empty = empty")
    } catch {
        expect(false, "Poly add test threw: \(error)")
    }
}

private func testPolySub(srs: [PointAffine]) {
    suite("GPUMarlinPolyIOP — Poly Sub")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)

        // [5, 7, 3] - [4, 5] = [1, 2, 3]
        let a = [frFromInt(5), frFromInt(7), frFromInt(3)]
        let b = [frFromInt(4), frFromInt(5)]
        let c = engine.polySub(a, b)
        expectEqual(c.count, 3, "Poly sub result length")
        expectEqual(c[0], frFromInt(1), "Poly sub coeff 0")
        expectEqual(c[1], frFromInt(2), "Poly sub coeff 1")
        expectEqual(c[2], frFromInt(3), "Poly sub coeff 2")

        // Self-subtraction = zero polynomial
        let d = engine.polySub(a, a)
        for i in 0..<d.count {
            expectEqual(d[i], Fr.zero, "Self-sub coeff \(i) is zero")
        }
    } catch {
        expect(false, "Poly sub test threw: \(error)")
    }
}

private func testPolyScalarMul(srs: [PointAffine]) {
    suite("GPUMarlinPolyIOP — Poly Scalar Mul")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)

        let a = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let s = frFromInt(5)
        let c = engine.polyScalarMul(a, s)
        expectEqual(c.count, 3, "Scalar mul result length")
        expectEqual(c[0], frFromInt(5), "Scalar mul coeff 0")
        expectEqual(c[1], frFromInt(10), "Scalar mul coeff 1")
        expectEqual(c[2], frFromInt(15), "Scalar mul coeff 2")

        // Multiply by zero
        let d = engine.polyScalarMul(a, .zero)
        for i in 0..<d.count {
            expectEqual(d[i], Fr.zero, "Zero scalar mul coeff \(i)")
        }

        // Multiply by one (identity)
        let e = engine.polyScalarMul(a, .one)
        for i in 0..<a.count {
            expectEqual(e[i], a[i], "One scalar mul coeff \(i)")
        }
    } catch {
        expect(false, "Poly scalar mul test threw: \(error)")
    }
}

private func testPolyMul(srs: [PointAffine]) {
    suite("GPUMarlinPolyIOP — Poly Mul (GPU NTT)")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)

        // (1 + 2x) * (3 + 4x) = 3 + 10x + 8x^2
        let a = [frFromInt(1), frFromInt(2)]
        let b = [frFromInt(3), frFromInt(4)]
        let c = try engine.polyMul(a, b)
        expectEqual(c.count, 3, "Poly mul result length")
        expectEqual(c[0], frFromInt(3), "Poly mul coeff 0")
        expectEqual(c[1], frFromInt(10), "Poly mul coeff 1")
        expectEqual(c[2], frFromInt(8), "Poly mul coeff 2")

        // (1) * (5 + 6x + 7x^2) = (5 + 6x + 7x^2)
        let d = [frFromInt(1)]
        let e = [frFromInt(5), frFromInt(6), frFromInt(7)]
        let f = try engine.polyMul(d, e)
        expectEqual(f.count, 3, "Identity mul length")
        expectEqual(f[0], frFromInt(5), "Identity mul coeff 0")
        expectEqual(f[1], frFromInt(6), "Identity mul coeff 1")
        expectEqual(f[2], frFromInt(7), "Identity mul coeff 2")
    } catch {
        expect(false, "Poly mul test threw: \(error)")
    }
}

private func testEvalVanishing(srs: [PointAffine]) {
    suite("GPUMarlinPolyIOP — Eval Vanishing")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)

        // v_D(omega) = omega^|D| - 1 should be 0 for any root of unity
        let logN = 3
        let domSize = 1 << logN
        let omega = frRootOfUnity(logN: logN)
        let vAtOmega = engine.evalVanishing(domainSize: domSize, at: omega)
        expectEqual(vAtOmega, Fr.zero, "Vanishing poly is zero at root of unity")

        // v_D(2) = 2^|D| - 1 (non-zero for random point)
        let x = frFromInt(2)
        let vAt2 = engine.evalVanishing(domainSize: domSize, at: x)
        let expected = frSub(frPow(x, UInt64(domSize)), .one)
        expectEqual(vAt2, expected, "Vanishing poly at x=2")
    } catch {
        expect(false, "Eval vanishing test threw: \(error)")
    }
}

private func testCPUEvalPoly(srs: [PointAffine]) {
    suite("GPUMarlinPolyIOP — CPU Eval Poly")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)

        // p(x) = 3 + 2x + x^2, p(5) = 3 + 10 + 25 = 38
        let coeffs = [frFromInt(3), frFromInt(2), frFromInt(1)]
        let result = engine.cpuEvalPoly(coeffs, at: frFromInt(5))
        expectEqual(result, frFromInt(38), "CPU eval poly: 3+2*5+5^2=38")

        // Zero polynomial
        let zero = engine.cpuEvalPoly([Fr.zero], at: frFromInt(100))
        expectEqual(zero, Fr.zero, "CPU eval zero poly")

        // Constant polynomial
        let c = engine.cpuEvalPoly([frFromInt(42)], at: frFromInt(999))
        expectEqual(c, frFromInt(42), "CPU eval constant poly")

        // Empty polynomial
        let e = engine.cpuEvalPoly([], at: frFromInt(1))
        expectEqual(e, Fr.zero, "CPU eval empty poly")
    } catch {
        expect(false, "CPU eval poly test threw: \(error)")
    }
}

private func testGPUEvalPoly(srs: [PointAffine]) {
    suite("GPUMarlinPolyIOP — GPU Eval Poly")

    do {
        // Use forceGPU to ensure GPU path is taken even for small polys
        let engine = try GPUMarlinPolyIOPEngine(srs: srs, config: .forceGPU)

        let coeffs = [frFromInt(3), frFromInt(2), frFromInt(1)]
        let result = engine.gpuEvalPoly(coeffs, at: frFromInt(5))
        let expected = frFromInt(38) // 3 + 2*5 + 5^2
        expectEqual(result, expected, "GPU eval poly: 3+2*5+5^2=38")

        // Force CPU path
        let cpuEngine = try GPUMarlinPolyIOPEngine(srs: srs, config: .forceCPU)
        let cpuResult = cpuEngine.gpuEvalPoly(coeffs, at: frFromInt(5))
        expectEqual(cpuResult, expected, "CPU fallback eval poly")
        expect(cpuEngine.lastStats.cpuEvalCount > 0, "CPU eval was used")
    } catch {
        expect(false, "GPU eval poly test threw: \(error)")
    }
}

// MARK: - Poseidon2 Hashing Tests

private func testHashPair(srs: [PointAffine]) {
    suite("GPUMarlinPolyIOP — Hash Pair")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)

        let a = frFromInt(1)
        let b = frFromInt(2)
        let h1 = engine.hashPair(a, b)
        let h2 = poseidon2Hash(a, b)
        expectEqual(h1, h2, "hashPair matches poseidon2Hash")

        // Different inputs give different hashes
        let h3 = engine.hashPair(b, a)
        expect(!frEqual(h1, h3), "hash(a,b) != hash(b,a)")
    } catch {
        expect(false, "Hash pair test threw: \(error)")
    }
}

private func testHashSequence(srs: [PointAffine]) {
    suite("GPUMarlinPolyIOP — Hash Sequence")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)

        // Single element
        let h1 = engine.hashSequence([frFromInt(42)])
        expectEqual(h1, frFromInt(42), "Hash single element returns itself")

        // Empty
        let h0 = engine.hashSequence([])
        expectEqual(h0, Fr.zero, "Hash empty returns zero")

        // Two elements = hashPair
        let a = frFromInt(1)
        let b = frFromInt(2)
        let h2 = engine.hashSequence([a, b])
        let expected = poseidon2Hash(a, b)
        expectEqual(h2, expected, "Hash two elements = hashPair")

        // Four elements = tree hash
        let elems = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let h4 = engine.hashSequence(elems)
        let left = poseidon2Hash(frFromInt(1), frFromInt(2))
        let right = poseidon2Hash(frFromInt(3), frFromInt(4))
        let root = poseidon2Hash(left, right)
        expectEqual(h4, root, "Hash four elements = Merkle tree")
    } catch {
        expect(false, "Hash sequence test threw: \(error)")
    }
}

// MARK: - Indexing Tests

private func testIndexSimple(srs: [PointAffine], srsSecret: Fr) {
    suite("GPUMarlinPolyIOP — Index Simple Circuit")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)

        // Simple multiply: a * b = c
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()
        aE.append(R1CSEntry(row: 0, col: 2, val: one))
        bE.append(R1CSEntry(row: 0, col: 3, val: one))
        cE.append(R1CSEntry(row: 0, col: 1, val: one))

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 1,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let indexed = try engine.indexR1CS(r1cs: r1cs, srsSecret: srsSecret)

        expectEqual(indexed.indexPolynomials.count, 12, "12 index polynomials (4 per matrix)")
        expectEqual(indexed.indexCommitments.count, 12, "12 index commitments")
        expectEqual(indexed.numPublic, 1, "1 public input")
        expectEqual(indexed.index.numConstraints, 1, "1 constraint")
        expectEqual(indexed.index.numVariables, 4, "4 variables")

        // Domain sizes are powers of 2 and >= counts
        expect(indexed.index.constraintDomainSize >= 1, "Constraint domain >= numConstraints")
        expect(indexed.index.variableDomainSize >= 4, "Variable domain >= numVars")
        expect(indexed.index.nonZeroDomainSize >= 1, "Non-zero domain >= maxNNZ")
    } catch {
        expect(false, "Index simple test threw: \(error)")
    }
}

private func testIndexMultiConstraint(srs: [PointAffine], srsSecret: Fr) {
    suite("GPUMarlinPolyIOP — Index Multi-Constraint")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)
        let one = Fr.one
        let numCons = 8
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        for i in 0..<numCons {
            aE.append(R1CSEntry(row: i, col: 1 + numCons + i, val: one))
            bE.append(R1CSEntry(row: i, col: 1 + numCons + i, val: one))
            cE.append(R1CSEntry(row: i, col: 1 + i, val: one))
        }

        let r1cs = R1CSInstance(numConstraints: numCons, numVars: 1 + 2 * numCons,
                                numPublic: numCons,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let indexed = try engine.indexR1CS(r1cs: r1cs, srsSecret: srsSecret)
        expectEqual(indexed.index.numConstraints, numCons, "\(numCons) constraints")
        expectEqual(indexed.indexPolynomials.count, 12, "Still 12 index polynomials")

        // Each polynomial should have degree = nonZeroDomainSize
        for poly in indexed.indexPolynomials {
            expectEqual(poly.count, indexed.index.nonZeroDomainSize,
                        "Index poly size = nonZeroDomainSize")
        }
    } catch {
        expect(false, "Index multi-constraint test threw: \(error)")
    }
}

// MARK: - Witness Construction Tests

private func testWitnessPolynomials(srs: [PointAffine], srsSecret: Fr) {
    suite("GPUMarlinPolyIOP — Witness Polynomials")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()
        aE.append(R1CSEntry(row: 0, col: 2, val: one))
        bE.append(R1CSEntry(row: 0, col: 3, val: one))
        cE.append(R1CSEntry(row: 0, col: 1, val: one))

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 1,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let indexed = try engine.indexR1CS(r1cs: r1cs, srsSecret: srsSecret)

        let aVal = frFromInt(3)
        let bVal = frFromInt(5)
        let cVal = frMul(aVal, bVal) // 15

        let witPoly = try engine.buildWitnessPolynomials(
            r1cs: r1cs, publicInputs: [cVal], witness: [aVal, bVal], indexed: indexed
        )

        // Full assignment should be [1, c, a, b]
        expectEqual(witPoly.fullAssignment[0], Fr.one, "z[0] = 1")
        expectEqual(witPoly.fullAssignment[1], cVal, "z[1] = c")
        expectEqual(witPoly.fullAssignment[2], aVal, "z[2] = a")
        expectEqual(witPoly.fullAssignment[3], bVal, "z[3] = b")

        // Coefficient arrays should be non-empty
        expect(witPoly.wCoeffs.count > 0, "w coefficients non-empty")
        expect(witPoly.zACoeffs.count > 0, "zA coefficients non-empty")
        expect(witPoly.zBCoeffs.count > 0, "zB coefficients non-empty")
        expect(witPoly.zCCoeffs.count > 0, "zC coefficients non-empty")

        // z_A(omega^0) should equal the first constraint's A*z value = a = 3
        // z_B(omega^0) should equal b = 5
        // z_C(omega^0) should equal c = 15
        // Verify A*z*B*z = C*z on first constraint
        let zAat0 = engine.cpuEvalPoly(witPoly.zACoeffs, at: .one)
        let zBat0 = engine.cpuEvalPoly(witPoly.zBCoeffs, at: .one)
        let zCat0 = engine.cpuEvalPoly(witPoly.zCCoeffs, at: .one)
        let product = frMul(zAat0, zBat0)
        // For satisfied R1CS on domain, z_A*z_B should equal z_C
        // (At omega^0 = 1, this corresponds to constraint 0)
        expectEqual(product, zCat0, "z_A(1)*z_B(1) = z_C(1) on satisfied circuit")
    } catch {
        expect(false, "Witness polynomials test threw: \(error)")
    }
}

// MARK: - Quotient Polynomial Tests

private func testQuotientPolynomial(srs: [PointAffine], srsSecret: Fr) {
    suite("GPUMarlinPolyIOP — Quotient Polynomial")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()
        aE.append(R1CSEntry(row: 0, col: 2, val: one))
        bE.append(R1CSEntry(row: 0, col: 3, val: one))
        cE.append(R1CSEntry(row: 0, col: 1, val: one))

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 1,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let indexed = try engine.indexR1CS(r1cs: r1cs, srsSecret: srsSecret)

        let aVal = frFromInt(3)
        let bVal = frFromInt(5)
        let cVal = frMul(aVal, bVal)

        let witPoly = try engine.buildWitnessPolynomials(
            r1cs: r1cs, publicInputs: [cVal], witness: [aVal, bVal], indexed: indexed
        )

        let hSize = indexed.index.constraintDomainSize
        let tCoeffs = try engine.buildQuotientPolynomial(witPoly: witPoly, hSize: hSize)

        // t(X) should have degree < |H|
        expectEqual(tCoeffs.count, hSize, "Quotient polynomial length = hSize")

        // Verify: z_A(x)*z_B(x) - z_C(x) = t(x) * v_H(x) at a random point
        let x = frFromInt(17)
        let zA_x = engine.cpuEvalPoly(witPoly.zACoeffs, at: x)
        let zB_x = engine.cpuEvalPoly(witPoly.zBCoeffs, at: x)
        let zC_x = engine.cpuEvalPoly(witPoly.zCCoeffs, at: x)
        let t_x = engine.cpuEvalPoly(tCoeffs, at: x)
        let vH_x = engine.evalVanishing(domainSize: hSize, at: x)

        let lhs = frSub(frMul(zA_x, zB_x), zC_x)
        let rhs = frMul(t_x, vH_x)
        expectEqual(lhs, rhs, "z_A*z_B - z_C = t * v_H at random point")
    } catch {
        expect(false, "Quotient polynomial test threw: \(error)")
    }
}

// MARK: - Sumcheck Tests

private func testSumcheckPolys(srs: [PointAffine]) {
    suite("GPUMarlinPolyIOP — Sumcheck Polynomials")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)
        let alpha = frFromInt(42)
        let state = engine.buildSumcheckPolys(3, alpha: alpha)

        expectEqual(state.roundPolynomials.count, 3, "3 sumcheck rounds")
        expectEqual(state.challenges.count, 3, "3 challenges")
        expectEqual(state.alpha, alpha, "Alpha preserved")

        // Each round polynomial has 3 coefficients (degree 2)
        for (i, poly) in state.roundPolynomials.enumerated() {
            expectEqual(poly.count, 3, "Round \(i) poly has 3 coefficients")
        }

        // First round: s_0(0) + s_0(1) = 0
        let s0 = state.roundPolynomials[0]
        let sum0 = frAdd(s0[0], s0[1])
        expectEqual(sum0, Fr.zero, "s_0(0) + s_0(1) = 0")
    } catch {
        expect(false, "Sumcheck polys test threw: \(error)")
    }
}

private func testSumcheckConsistency(srs: [PointAffine]) {
    suite("GPUMarlinPolyIOP — Sumcheck Consistency")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)
        let alpha = frFromInt(123)
        let state = engine.buildSumcheckPolys(5, alpha: alpha)

        // Verify chain property: s_{i+1}(0) + s_{i+1}(1) = s_i(r_i)
        for r in 1..<state.roundPolynomials.count {
            let nextPoly = state.roundPolynomials[r]
            let nextSum = frAdd(nextPoly[0], nextPoly[1])

            // Evaluate previous round poly at its challenge
            let prevPoly = state.roundPolynomials[r - 1]
            let ri = state.challenges[r - 1]

            // Lagrange interpolation on {0,1,2}
            let f0 = prevPoly[0], f1 = prevPoly[1], f2 = prevPoly[2]
            let rM1 = frSub(ri, .one)
            let rM2 = frSub(ri, frFromInt(2))
            let inv2 = frInverse(frFromInt(2))
            let t0 = frMul(f0, frMul(frMul(rM1, rM2), inv2))
            let t1 = frMul(frNeg(f1), frMul(ri, rM2))
            let t2 = frMul(f2, frMul(frMul(ri, rM1), inv2))
            let prevAtR = frAdd(frAdd(t0, t1), t2)

            expectEqual(nextSum, prevAtR,
                        "Sumcheck chain: s_\(r)(0)+s_\(r)(1) = s_\(r-1)(r_\(r-1))")
        }
    } catch {
        expect(false, "Sumcheck consistency test threw: \(error)")
    }
}

// MARK: - Holographic Reduction Tests

private func testHolographicReduction(srs: [PointAffine], srsSecret: Fr) {
    suite("GPUMarlinPolyIOP — Holographic Reduction")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()
        aE.append(R1CSEntry(row: 0, col: 2, val: one))
        bE.append(R1CSEntry(row: 0, col: 3, val: one))
        cE.append(R1CSEntry(row: 0, col: 1, val: one))

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 1,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let indexed = try engine.indexR1CS(r1cs: r1cs, srsSecret: srsSecret)

        let etas = [frFromInt(7), frFromInt(11), frFromInt(13)]
        let beta = frFromInt(97)

        let holo = try engine.buildHolographicReduction(
            indexed: indexed, etas: etas, beta: beta
        )

        let nzSize = indexed.index.nonZeroDomainSize
        expectEqual(holo.sigmaEvals.count, nzSize, "Sigma evals count = nzSize")
        expectEqual(holo.gCoeffs.count, nzSize, "g polynomial size = nzSize")
        expect(holo.hCoeffs.count >= 2, "h polynomial has at least 2 coeffs")

        // g should interpolate sigma: g(omega^i) = sigma[i] for each i
        let logNZ = Int(log2(Double(nzSize)))
        let omegaNZ = frRootOfUnity(logN: logNZ)
        for i in 0..<min(nzSize, 4) {
            let pt = frPow(omegaNZ, UInt64(i))
            let gAtPt = engine.cpuEvalPoly(holo.gCoeffs, at: pt)
            expectEqual(gAtPt, holo.sigmaEvals[i],
                        "g(omega^\(i)) = sigma[\(i)]")
        }
    } catch {
        expect(false, "Holographic reduction test threw: \(error)")
    }
}

// MARK: - Full Prove + Verify Tests

private func testProveVerifyMultiply(srs: [PointAffine], srsSecret: Fr) {
    suite("GPUMarlinPolyIOP — Prove+Verify Multiply")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()
        aE.append(R1CSEntry(row: 0, col: 2, val: one))
        bE.append(R1CSEntry(row: 0, col: 3, val: one))
        cE.append(R1CSEntry(row: 0, col: 1, val: one))

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 1,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let aVal = frFromInt(3)
        let bVal = frFromInt(5)
        let cVal = frMul(aVal, bVal) // 15

        // Verify R1CS is satisfied
        var z = [Fr](repeating: .zero, count: 4)
        z[0] = .one; z[1] = cVal; z[2] = aVal; z[3] = bVal
        expect(r1cs.isSatisfied(z: z), "Multiply R1CS satisfied")

        let indexed = try engine.indexR1CS(r1cs: r1cs, srsSecret: srsSecret)
        let proof = try engine.prove(
            r1cs: r1cs, publicInputs: [cVal], witness: [aVal, bVal], indexed: indexed
        )

        let valid = engine.verify(indexed: indexed, publicInput: [cVal], proof: proof)
        expect(valid, "Multiply circuit GPU proof verifies")
    } catch {
        expect(false, "Prove+verify multiply threw: \(error)")
    }
}

private func testProveVerify10Constraints(srs: [PointAffine], srsSecret: Fr) {
    suite("GPUMarlinPolyIOP — Prove+Verify 10 Constraints")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)
        let one = Fr.one
        let numCons = 10
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()

        for i in 0..<numCons {
            aE.append(R1CSEntry(row: i, col: 1 + numCons + i, val: one))
            bE.append(R1CSEntry(row: i, col: 1 + numCons + i, val: one))
            cE.append(R1CSEntry(row: i, col: 1 + i, val: one))
        }

        let r1cs = R1CSInstance(numConstraints: numCons, numVars: 1 + 2 * numCons,
                                numPublic: numCons,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        var publicInputs = [Fr]()
        var witness = [Fr]()
        for i in 0..<numCons {
            let x = frFromInt(UInt64(i + 2))
            let y = frMul(x, x)
            publicInputs.append(y)
            witness.append(x)
        }

        var zFull = [Fr](repeating: .zero, count: 1 + 2 * numCons)
        zFull[0] = .one
        for i in 0..<numCons { zFull[1 + i] = publicInputs[i] }
        for i in 0..<numCons { zFull[1 + numCons + i] = witness[i] }
        expect(r1cs.isSatisfied(z: zFull), "10-constraint R1CS satisfied")

        let indexed = try engine.indexR1CS(r1cs: r1cs, srsSecret: srsSecret)
        let proof = try engine.prove(
            r1cs: r1cs, publicInputs: publicInputs, witness: witness, indexed: indexed
        )

        let valid = engine.verify(indexed: indexed, publicInput: publicInputs, proof: proof)
        expect(valid, "10-constraint GPU proof verifies")
    } catch {
        expect(false, "Prove+verify 10 constraints threw: \(error)")
    }
}

private func testWrongWitnessRejected(srs: [PointAffine], srsSecret: Fr) {
    suite("GPUMarlinPolyIOP — Wrong Witness Rejected")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()
        aE.append(R1CSEntry(row: 0, col: 2, val: one))
        bE.append(R1CSEntry(row: 0, col: 3, val: one))
        cE.append(R1CSEntry(row: 0, col: 1, val: one))

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 1,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        // Wrong witness: claim c=15 but a=3, b=4 (3*4=12, not 15)
        let cVal = frFromInt(15)
        let aVal = frFromInt(3)
        let bVal = frFromInt(4)

        var z = [Fr](repeating: .zero, count: 4)
        z[0] = .one; z[1] = cVal; z[2] = aVal; z[3] = bVal
        expect(!r1cs.isSatisfied(z: z), "Wrong witness does not satisfy R1CS")

        let indexed = try engine.indexR1CS(r1cs: r1cs, srsSecret: srsSecret)
        let proof = try engine.prove(
            r1cs: r1cs, publicInputs: [cVal], witness: [aVal, bVal], indexed: indexed
        )

        let valid = engine.verify(indexed: indexed, publicInput: [cVal], proof: proof)
        expect(!valid, "Wrong witness GPU proof is rejected")
    } catch {
        // Throwing during prove with bad witness is also acceptable rejection
        expect(true, "Wrong witness rejected via exception")
    }
}

private func testIndexReuse(srs: [PointAffine], srsSecret: Fr) {
    suite("GPUMarlinPolyIOP — Index Reuse")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()
        aE.append(R1CSEntry(row: 0, col: 2, val: one))
        bE.append(R1CSEntry(row: 0, col: 3, val: one))
        cE.append(R1CSEntry(row: 0, col: 1, val: one))

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 1,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        // Index once, prove multiple times
        let indexed = try engine.indexR1CS(r1cs: r1cs, srsSecret: srsSecret)

        // Instance 1: 3 * 7 = 21
        let a1 = frFromInt(3), b1 = frFromInt(7)
        let c1 = frMul(a1, b1)
        let proof1 = try engine.prove(
            r1cs: r1cs, publicInputs: [c1], witness: [a1, b1], indexed: indexed
        )
        let valid1 = engine.verify(indexed: indexed, publicInput: [c1], proof: proof1)
        expect(valid1, "Index reuse: 3*7=21 verifies")

        // Instance 2: 5 * 11 = 55
        let a2 = frFromInt(5), b2 = frFromInt(11)
        let c2 = frMul(a2, b2)
        let proof2 = try engine.prove(
            r1cs: r1cs, publicInputs: [c2], witness: [a2, b2], indexed: indexed
        )
        let valid2 = engine.verify(indexed: indexed, publicInput: [c2], proof: proof2)
        expect(valid2, "Index reuse: 5*11=55 verifies")

        // Instance 3: 100 * 200 = 20000
        let a3 = frFromInt(100), b3 = frFromInt(200)
        let c3 = frMul(a3, b3)
        let proof3 = try engine.prove(
            r1cs: r1cs, publicInputs: [c3], witness: [a3, b3], indexed: indexed
        )
        let valid3 = engine.verify(indexed: indexed, publicInput: [c3], proof: proof3)
        expect(valid3, "Index reuse: 100*200=20000 verifies")

        // Cross-check: proof1 should NOT verify against instance 2's public input
        let crossValid = engine.verify(indexed: indexed, publicInput: [c2], proof: proof1)
        expect(!crossValid, "Cross-instance proof rejected")
    } catch {
        expect(false, "Index reuse test threw: \(error)")
    }
}

// MARK: - GPU vs CPU Equivalence Tests

private func testGPUvsCPUEquivalence(srs: [PointAffine], srsSecret: Fr) {
    suite("GPUMarlinPolyIOP — GPU vs CPU Equivalence")

    do {
        let gpuEngine = try GPUMarlinPolyIOPEngine(srs: srs, config: .forceGPU)
        let cpuEngine = try GPUMarlinPolyIOPEngine(srs: srs, config: .forceCPU)

        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()
        aE.append(R1CSEntry(row: 0, col: 2, val: one))
        bE.append(R1CSEntry(row: 0, col: 3, val: one))
        cE.append(R1CSEntry(row: 0, col: 1, val: one))

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 1,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let aVal = frFromInt(7)
        let bVal = frFromInt(11)
        let cVal = frMul(aVal, bVal) // 77

        // Index on both engines
        let gpuIndexed = try gpuEngine.indexR1CS(r1cs: r1cs, srsSecret: srsSecret)
        let cpuIndexed = try cpuEngine.indexR1CS(r1cs: r1cs, srsSecret: srsSecret)

        // Index polynomials should be identical
        for i in 0..<12 {
            for j in 0..<gpuIndexed.indexPolynomials[i].count {
                expectEqual(gpuIndexed.indexPolynomials[i][j],
                            cpuIndexed.indexPolynomials[i][j],
                            "Index poly[\(i)][\(j)] GPU == CPU")
            }
        }

        // Witness polynomials should be identical
        let gpuWit = try gpuEngine.buildWitnessPolynomials(
            r1cs: r1cs, publicInputs: [cVal], witness: [aVal, bVal], indexed: gpuIndexed
        )
        let cpuWit = try cpuEngine.buildWitnessPolynomials(
            r1cs: r1cs, publicInputs: [cVal], witness: [aVal, bVal], indexed: cpuIndexed
        )

        for i in 0..<gpuWit.zACoeffs.count {
            expectEqual(gpuWit.zACoeffs[i], cpuWit.zACoeffs[i], "zA[\(i)] GPU == CPU")
        }

        // Full prove on both, both should verify
        let gpuProof = try gpuEngine.prove(
            r1cs: r1cs, publicInputs: [cVal], witness: [aVal, bVal], indexed: gpuIndexed
        )
        let cpuProof = try cpuEngine.prove(
            r1cs: r1cs, publicInputs: [cVal], witness: [aVal, bVal], indexed: cpuIndexed
        )

        let gpuValid = gpuEngine.verify(indexed: gpuIndexed, publicInput: [cVal], proof: gpuProof)
        let cpuValid = cpuEngine.verify(indexed: cpuIndexed, publicInput: [cVal], proof: cpuProof)
        expect(gpuValid, "GPU proof verifies")
        expect(cpuValid, "CPU proof verifies")

        // Evaluations should be identical (same polynomial, same challenge points)
        expectEqual(gpuProof.evaluations.zABeta, cpuProof.evaluations.zABeta,
                    "z_A(beta) GPU == CPU")
        expectEqual(gpuProof.evaluations.zBBeta, cpuProof.evaluations.zBBeta,
                    "z_B(beta) GPU == CPU")
        expectEqual(gpuProof.evaluations.zCBeta, cpuProof.evaluations.zCBeta,
                    "z_C(beta) GPU == CPU")
        expectEqual(gpuProof.evaluations.wBeta, cpuProof.evaluations.wBeta,
                    "w(beta) GPU == CPU")
        expectEqual(gpuProof.evaluations.tBeta, cpuProof.evaluations.tBeta,
                    "t(beta) GPU == CPU")
    } catch {
        expect(false, "GPU vs CPU equivalence test threw: \(error)")
    }
}

// MARK: - Statistics Tests

private func testStatisticsTracking(srs: [PointAffine], srsSecret: Fr) {
    suite("GPUMarlinPolyIOP — Statistics Tracking")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()
        aE.append(R1CSEntry(row: 0, col: 2, val: one))
        bE.append(R1CSEntry(row: 0, col: 3, val: one))
        cE.append(R1CSEntry(row: 0, col: 1, val: one))

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 1,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let aVal = frFromInt(3), bVal = frFromInt(5)
        let cVal = frMul(aVal, bVal)

        let indexed = try engine.indexR1CS(r1cs: r1cs, srsSecret: srsSecret)
        _ = try engine.prove(
            r1cs: r1cs, publicInputs: [cVal], witness: [aVal, bVal], indexed: indexed
        )

        let stats = engine.lastStats
        // After a prove, there should be some operations tracked
        let totalEvals = stats.gpuEvalCount + stats.cpuEvalCount
        expect(totalEvals > 0, "Some polynomial evaluations were tracked")
        expect(stats.gpuNTTCount > 0, "NTT operations were tracked")
        expect(stats.totalTime > 0, "Total time was tracked")
        expect(stats.nttTime >= 0, "NTT time >= 0")
        expect(stats.commitTime >= 0, "Commit time >= 0")
        expect(stats.sigmaTime >= 0, "Sigma time >= 0")

        // Summary string should be non-empty
        let summary = stats.summary
        expect(summary.contains("GPUMarlinPolyIOP"), "Summary contains engine name")
        expect(summary.contains("GPU ops"), "Summary contains GPU ops count")
    } catch {
        expect(false, "Statistics tracking test threw: \(error)")
    }
}

// MARK: - Verify Diagnostics Tests

private func testVerifyDiag(srs: [PointAffine], srsSecret: Fr) {
    suite("GPUMarlinPolyIOP — Verify Diagnostics")

    do {
        let engine = try GPUMarlinPolyIOPEngine(srs: srs)
        let one = Fr.one
        var aE = [R1CSEntry](), bE = [R1CSEntry](), cE = [R1CSEntry]()
        aE.append(R1CSEntry(row: 0, col: 2, val: one))
        bE.append(R1CSEntry(row: 0, col: 3, val: one))
        cE.append(R1CSEntry(row: 0, col: 1, val: one))

        let r1cs = R1CSInstance(numConstraints: 1, numVars: 4, numPublic: 1,
                                aEntries: aE, bEntries: bE, cEntries: cE)

        let aVal = frFromInt(3), bVal = frFromInt(5)
        let cVal = frMul(aVal, bVal)

        let indexed = try engine.indexR1CS(r1cs: r1cs, srsSecret: srsSecret)
        let proof = try engine.prove(
            r1cs: r1cs, publicInputs: [cVal], witness: [aVal, bVal], indexed: indexed
        )

        let diag = engine.verifyDiag(indexed: indexed, publicInput: [cVal], proof: proof)
        expectEqual(diag, "PASS", "Diagnostic returns PASS for valid proof")
    } catch {
        expect(false, "Verify diagnostics test threw: \(error)")
    }
}
