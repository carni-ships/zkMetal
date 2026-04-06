import zkMetal

// MARK: - GPU Proof Composition Tests

public func runGPUProofCompositionTests() {
    suite("GPU Proof Composition Engine")

    let engine = GPUProofCompositionEngine()

    // Test 1: Linear combination — single polynomial scaled by 1 is identity
    do {
        let poly = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let result = engine.linearCombination(polys: [poly], challenges: [Fr.one])
        for i in 0..<poly.count {
            expect(frEqual(result[i], poly[i]),
                   "Linear combination: identity (coeff \(i))")
        }
    }

    // Test 2: Linear combination — two polynomials
    do {
        let p0 = [frFromInt(1), frFromInt(0), frFromInt(3)]
        let p1 = [frFromInt(0), frFromInt(2), frFromInt(0)]
        let c0 = frFromInt(5)
        let c1 = frFromInt(7)
        let result = engine.linearCombination(polys: [p0, p1], challenges: [c0, c1])

        // result[0] = 5*1 + 7*0 = 5
        expect(frEqual(result[0], frFromInt(5)), "LinComb 2-poly: coeff 0")
        // result[1] = 5*0 + 7*2 = 14
        expect(frEqual(result[1], frFromInt(14)), "LinComb 2-poly: coeff 1")
        // result[2] = 5*3 + 7*0 = 15
        expect(frEqual(result[2], frFromInt(15)), "LinComb 2-poly: coeff 2")
    }

    // Test 3: Linear combination — zero challenge zeroes out
    do {
        let poly = [frFromInt(10), frFromInt(20)]
        let result = engine.linearCombination(polys: [poly], challenges: [Fr.zero])
        expect(frEqual(result[0], Fr.zero), "LinComb zero challenge: coeff 0")
        expect(frEqual(result[1], Fr.zero), "LinComb zero challenge: coeff 1")
    }

    // Test 4: Batch opening — combined eval consistency
    do {
        // Two polynomials evaluated at z=2
        let p0 = [frFromInt(1), frFromInt(3)]  // 1 + 3x -> p0(2) = 7
        let p1 = [frFromInt(2), frFromInt(1)]  // 2 + x  -> p1(2) = 4
        let z = frFromInt(2)
        let v0 = engine.evaluatePolynomial(p0, at: z)
        let v1 = engine.evaluatePolynomial(p1, at: z)

        expect(frEqual(v0, frFromInt(7)), "Batch opening: p0(2) = 7")
        expect(frEqual(v1, frFromInt(4)), "Batch opening: p1(2) = 4")

        let r0 = frFromInt(3)
        let r1 = frFromInt(5)
        let (combinedPoly, combinedEval) = engine.batchOpening(
            polys: [p0, p1], evals: [v0, v1], challenges: [r0, r1])

        // combinedEval = 3*7 + 5*4 = 21 + 20 = 41
        expect(frEqual(combinedEval, frFromInt(41)), "Batch opening: combined eval = 41")

        // Verify: evaluate combined poly at z should equal combinedEval
        let evalCheck = engine.evaluatePolynomial(combinedPoly, at: z)
        expect(frEqual(evalCheck, combinedEval),
               "Batch opening: combined poly eval matches combined eval")
    }

    // Test 5: Accumulator fold — fold identity
    do {
        let poly = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let acc = ProofAccumulator.initial(poly: poly)
        expectEqual(acc.foldCount, 1, "Initial accumulator fold count = 1")
        expect(frEqual(acc.error, Fr.zero), "Initial accumulator error = 0")

        // Fold zero polynomial should not change acc
        let zeroPoly = [Fr](repeating: Fr.zero, count: 4)
        let acc2 = engine.fold(accumulator: acc, newPoly: zeroPoly, challenge: frFromInt(99))
        for i in 0..<4 {
            expect(frEqual(acc2.accPoly[i], poly[i]),
                   "Fold zero poly: accPoly unchanged (coeff \(i))")
        }
        expectEqual(acc2.foldCount, 2, "After fold: count = 2")
    }

    // Test 6: Accumulator fold — algebraic correctness
    do {
        let acc0 = [frFromInt(1), frFromInt(2)]
        let newP = [frFromInt(3), frFromInt(4)]
        let r = frFromInt(5)

        let acc = ProofAccumulator.initial(poly: acc0)
        let folded = engine.fold(accumulator: acc, newPoly: newP, challenge: r)

        // acc' = [1 + 5*3, 2 + 5*4] = [16, 22]
        expect(frEqual(folded.accPoly[0], frFromInt(16)),
               "Fold algebraic: acc'[0] = 16")
        expect(frEqual(folded.accPoly[1], frFromInt(22)),
               "Fold algebraic: acc'[1] = 22")

        // cross = 1*3 + 2*4 = 11, error' = 0 + 25*11 = 275
        expect(frEqual(folded.error, frFromInt(275)),
               "Fold algebraic: error = 275")
        expectEqual(folded.foldCount, 2, "Fold algebraic: count = 2")
    }

    // Test 7: Challenge derivation — determinism
    do {
        let transcript = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let c1 = engine.deriveChallenge(transcript: transcript, domainSeparator: "test-step")
        let c2 = engine.deriveChallenge(transcript: transcript, domainSeparator: "test-step")
        expect(frEqual(c1, c2), "Challenge derivation: deterministic (same input -> same output)")

        // Different domain separator -> different challenge
        let c3 = engine.deriveChallenge(transcript: transcript, domainSeparator: "other-step")
        expect(!frEqual(c1, c3), "Challenge derivation: different domain -> different challenge")

        // Different transcript -> different challenge
        let transcript2 = [frFromInt(1), frFromInt(2), frFromInt(4)]
        let c4 = engine.deriveChallenge(transcript: transcript2, domainSeparator: "test-step")
        expect(!frEqual(c1, c4), "Challenge derivation: different transcript -> different challenge")
    }

    // Test 8: Challenge is nonzero (overwhelmingly likely)
    do {
        let transcript = [frFromInt(42)]
        let c = engine.deriveChallenge(transcript: transcript, domainSeparator: "nonzero-check")
        expect(!frEqual(c, Fr.zero), "Challenge derivation: output is nonzero")
    }

    // Test 9: Multi-point combination — gamma powers are correct
    do {
        let poly = [frFromInt(10), frFromInt(20)]
        let points = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let values = [frFromInt(30), frFromInt(50), frFromInt(70)]
        let gamma = frFromInt(2)

        let (numerators, gammaFactors) = engine.multiPointCombination(
            poly: poly, points: points, values: values, gamma: gamma)

        // gamma^0 = 1, gamma^1 = 2, gamma^2 = 4
        expect(frEqual(gammaFactors[0], Fr.one), "Multi-point: gamma^0 = 1")
        expect(frEqual(gammaFactors[1], frFromInt(2)), "Multi-point: gamma^1 = 2")
        expect(frEqual(gammaFactors[2], frFromInt(4)), "Multi-point: gamma^2 = 4")

        expectEqual(numerators.count, 3, "Multi-point: 3 numerator polys")
    }

    // Test 10: Multi-point combination — numerator structure
    do {
        // poly = [5, 3] means 5 + 3x
        // v_0 = 8, so numerator_0 = poly - 8 = [-3, 3], scaled by gamma^0 = 1 -> [-3, 3]
        let poly = [frFromInt(5), frFromInt(3)]
        let points = [frFromInt(1)]
        let values = [frFromInt(8)]
        let gamma = frFromInt(1)

        let (numerators, _) = engine.multiPointCombination(
            poly: poly, points: points, values: values, gamma: gamma)

        // numerator[0] = gamma^0 * (poly - 8) = [5-8, 3] = [-3, 3]
        let negThree = frSub(frFromInt(5), frFromInt(8))
        expect(frEqual(numerators[0][0], negThree), "Multi-point numerator: constant term")
        expect(frEqual(numerators[0][1], frFromInt(3)), "Multi-point numerator: linear term")
    }

    // Test 11: Polynomial evaluation via Horner
    do {
        // p(x) = 1 + 2x + 3x^2, p(2) = 1 + 4 + 12 = 17
        let poly = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let val = engine.evaluatePolynomial(poly, at: frFromInt(2))
        expect(frEqual(val, frFromInt(17)), "Horner eval: 1+2*2+3*4 = 17")
    }

    // Test 12: Polynomial evaluation — empty and constant
    do {
        let empty = engine.evaluatePolynomial([], at: frFromInt(5))
        expect(frEqual(empty, Fr.zero), "Horner eval: empty poly = 0")

        let constant = engine.evaluatePolynomial([frFromInt(42)], at: frFromInt(999))
        expect(frEqual(constant, frFromInt(42)), "Horner eval: constant poly")
    }

    // Test 13: CompositionTranscript — squeeze determinism
    do {
        var t1 = CompositionTranscript()
        t1.appendLabel("test")
        t1.appendScalar(frFromInt(7))
        let c1 = t1.challenge()

        var t2 = CompositionTranscript()
        t2.appendLabel("test")
        t2.appendScalar(frFromInt(7))
        let c2 = t2.challenge()

        expect(frEqual(c1, c2), "CompositionTranscript: deterministic squeeze")
    }

    // Test 14: Multiple folds accumulate challenges
    do {
        let poly = [frFromInt(1), frFromInt(1)]
        var acc = ProofAccumulator.initial(poly: poly)

        for i in 1...3 {
            let newP = [frFromInt(UInt64(i)), frFromInt(UInt64(i))]
            let r = frFromInt(UInt64(10 + i))
            acc = engine.fold(accumulator: acc, newPoly: newP, challenge: r)
        }

        expectEqual(acc.foldCount, 4, "Multiple folds: count = 4")
        expectEqual(acc.challenges.count, 3, "Multiple folds: 3 challenges recorded")
    }

    print("  GPU Proof Composition engine version: \(GPUProofCompositionEngine.version.description)")
}
