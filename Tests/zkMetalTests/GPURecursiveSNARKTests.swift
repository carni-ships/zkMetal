import zkMetal

// MARK: - GPU Recursive SNARK Engine Tests

public func runGPURecursiveSNARKTests() {
    suite("GPU Recursive SNARK Engine")

    let engine = GPURecursiveSNARKEngine()

    // Test 1: Single accumulation — fold one polynomial into a zero accumulator
    do {
        let poly = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let acc0 = RecursiveAccumulator.zero(size: 4)
        let challenge = frFromInt(7)
        let acc1 = engine.fold(accumulator: acc0, newPoly: poly, challenge: challenge)

        // acc' = 0 + 7 * poly = [7, 14, 21, 28]
        expect(frEqual(acc1.accPoly[0], frFromInt(7)), "Single accumulation: coeff 0 = 7")
        expect(frEqual(acc1.accPoly[1], frFromInt(14)), "Single accumulation: coeff 1 = 14")
        expect(frEqual(acc1.accPoly[2], frFromInt(21)), "Single accumulation: coeff 2 = 21")
        expect(frEqual(acc1.accPoly[3], frFromInt(28)), "Single accumulation: coeff 3 = 28")
        expectEqual(acc1.foldCount, 1, "Single accumulation: fold count = 1")
        // Cross term = <0, poly> = 0, so error = 0 + 49 * 0 = 0
        expect(frEqual(acc1.error, Fr.zero), "Single accumulation: error = 0 (zero cross term)")
        expectEqual(acc1.challenges.count, 1, "Single accumulation: 1 challenge recorded")
    }

    // Test 2: Double fold — fold two polynomials sequentially
    do {
        let p1 = [frFromInt(1), frFromInt(2)]
        let p2 = [frFromInt(3), frFromInt(4)]
        let acc0 = RecursiveAccumulator.initial(poly: p1)

        let challenge = frFromInt(5)
        let acc1 = engine.fold(accumulator: acc0, newPoly: p2, challenge: challenge)

        // acc' = [1, 2] + 5 * [3, 4] = [1+15, 2+20] = [16, 22]
        expect(frEqual(acc1.accPoly[0], frFromInt(16)), "Double fold: coeff 0 = 16")
        expect(frEqual(acc1.accPoly[1], frFromInt(22)), "Double fold: coeff 1 = 22")
        expectEqual(acc1.foldCount, 2, "Double fold: fold count = 2")

        // Cross term = <[1,2], [3,4]> = 1*3 + 2*4 = 11
        // error' = 0 + 25 * 11 = 275
        expect(frEqual(acc1.error, frFromInt(275)), "Double fold: error = 275")
        expectEqual(acc1.challenges.count, 1, "Double fold: 1 challenge recorded")

        // Fold a third polynomial
        let p3 = [frFromInt(1), frFromInt(1)]
        let challenge2 = frFromInt(2)
        let acc2 = engine.fold(accumulator: acc1, newPoly: p3, challenge: challenge2)

        // acc'' = [16, 22] + 2 * [1, 1] = [18, 24]
        expect(frEqual(acc2.accPoly[0], frFromInt(18)), "Triple fold: coeff 0 = 18")
        expect(frEqual(acc2.accPoly[1], frFromInt(24)), "Triple fold: coeff 1 = 24")
        expectEqual(acc2.foldCount, 3, "Triple fold: fold count = 3")

        // Cross term = <[16, 22], [1, 1]> = 16 + 22 = 38
        // error'' = 275 + 4 * 38 = 275 + 152 = 427
        expect(frEqual(acc2.error, frFromInt(427)), "Triple fold: error = 427")
    }

    // Test 3: Deferred verification — pairing checks
    do {
        var acc = RecursiveAccumulator.zero(size: 2)

        // Add a balanced pairing check: lhsG1 * lhsG2 == rhsG1 * rhsG2
        // 3 * 7 = 21 and 21 * 1 = 21 -> should pass
        let check1 = DeferredPairingCheck(
            lhsG1: frFromInt(3), lhsG2: frFromInt(7),
            rhsG1: frFromInt(21), rhsG2: Fr.one
        )
        acc = engine.accumulatePairingCheck(accumulator: acc, check: check1)
        expectEqual(acc.deferredPairings.count, 1, "Deferred verify: 1 pairing check")
        expectEqual(acc.totalDeferredChecks, 1, "Deferred verify: total = 1")

        // Verify balanced check passes
        var transcript = RecursiveTranscript()
        let ok = engine.verifyDeferredPairings(accumulator: acc, transcript: &transcript)
        expect(ok, "Deferred verify: balanced pairing check passes")

        // Add an unbalanced check: 2 * 3 = 6 != 5 * 2 = 10
        let check2 = DeferredPairingCheck(
            lhsG1: frFromInt(2), lhsG2: frFromInt(3),
            rhsG1: frFromInt(5), rhsG2: frFromInt(2)
        )
        acc = engine.accumulatePairingCheck(accumulator: acc, check: check2)
        expectEqual(acc.deferredPairings.count, 2, "Deferred verify: 2 pairing checks")

        var transcript2 = RecursiveTranscript()
        let ok2 = engine.verifyDeferredPairings(accumulator: acc, transcript: &transcript2)
        expect(!ok2, "Deferred verify: unbalanced pairing check fails")
    }

    // Test 4: Hash transcript consistency
    do {
        // Same inputs should produce same challenge
        var t1 = RecursiveTranscript()
        t1.appendLabel("test-domain")
        t1.appendScalar(frFromInt(42))
        let c1 = t1.squeeze()

        var t2 = RecursiveTranscript()
        t2.appendLabel("test-domain")
        t2.appendScalar(frFromInt(42))
        let c2 = t2.squeeze()

        expect(frEqual(c1, c2), "Transcript consistency: same inputs -> same challenge")

        // Different inputs should produce different challenge
        var t3 = RecursiveTranscript()
        t3.appendLabel("test-domain")
        t3.appendScalar(frFromInt(43))
        let c3 = t3.squeeze()

        expect(!frEqual(c1, c3), "Transcript consistency: different inputs -> different challenge")

        // State hash should also be deterministic
        let h1 = t1.stateHash()
        let h2 = t2.stateHash()
        expect(h1 == h2, "Transcript consistency: state hashes match")

        // squeezeAndAdvance should advance state
        var t4 = RecursiveTranscript()
        t4.appendLabel("advance-test")
        let ca = t4.squeezeAndAdvance()
        let cb = t4.squeeze()
        expect(!frEqual(ca, cb), "Transcript consistency: squeezeAndAdvance changes state")
    }

    // Test 5: Accumulator reset
    do {
        let poly = [frFromInt(10), frFromInt(20)]
        var acc = RecursiveAccumulator.initial(poly: poly)
        let check = DeferredPairingCheck(
            lhsG1: frFromInt(1), lhsG2: frFromInt(1),
            rhsG1: frFromInt(1), rhsG2: frFromInt(1)
        )
        acc = engine.accumulatePairingCheck(accumulator: acc, check: check)
        let ipaCheck = DeferredIPACheck(
            commitment: frFromInt(5), point: frFromInt(2), value: frFromInt(3),
            lPoints: [frFromInt(1)], rPoints: [frFromInt(2)]
        )
        acc = engine.accumulateIPACheck(accumulator: acc, check: ipaCheck)

        expectEqual(acc.foldCount, 1, "Before reset: fold count = 1")
        expectEqual(acc.deferredPairings.count, 1, "Before reset: 1 pairing check")
        expectEqual(acc.deferredIPAs.count, 1, "Before reset: 1 IPA check")

        acc.reset()

        expectEqual(acc.foldCount, 0, "After reset: fold count = 0")
        expect(frEqual(acc.error, Fr.zero), "After reset: error = 0")
        expectEqual(acc.deferredPairings.count, 0, "After reset: 0 pairing checks")
        expectEqual(acc.deferredIPAs.count, 0, "After reset: 0 IPA checks")
        expectEqual(acc.challenges.count, 0, "After reset: 0 challenges")
        expectEqual(acc.accPoly.count, 2, "After reset: poly size preserved")
        expect(frEqual(acc.accPoly[0], Fr.zero), "After reset: poly[0] = 0")
        expect(frEqual(acc.accPoly[1], Fr.zero), "After reset: poly[1] = 0")
    }

    // Test 6: Full fold with deferred verification (KZG path)
    do {
        let initPoly = [frFromInt(1), frFromInt(2)]
        var acc = RecursiveAccumulator.initial(poly: initPoly)
        var transcript = RecursiveTranscript()

        let innerPoly = [frFromInt(3), frFromInt(4)]
        let pairingCheck = DeferredPairingCheck(
            lhsG1: frFromInt(6), lhsG2: Fr.one,
            rhsG1: frFromInt(6), rhsG2: Fr.one
        )
        acc = engine.foldWithDeferredVerification(
            accumulator: acc,
            innerPoly: innerPoly,
            pcsType: .kzg,
            pairingCheck: pairingCheck,
            transcript: &transcript
        )

        expectEqual(acc.foldCount, 2, "Full fold KZG: fold count = 2")
        expectEqual(acc.deferredPairings.count, 1, "Full fold KZG: 1 deferred pairing")
        expectEqual(acc.challenges.count, 1, "Full fold KZG: 1 challenge")
    }

    // Test 7: Full fold with deferred verification (IPA path)
    do {
        let initPoly = [frFromInt(5), frFromInt(6)]
        var acc = RecursiveAccumulator.initial(poly: initPoly)
        var transcript = RecursiveTranscript()

        let innerPoly = [frFromInt(7), frFromInt(8)]
        let ipaCheck = DeferredIPACheck(
            commitment: frFromInt(10), point: frFromInt(2), value: frFromInt(5),
            lPoints: [frFromInt(1), frFromInt(2)],
            rPoints: [frFromInt(3), frFromInt(4)]
        )
        acc = engine.foldWithDeferredVerification(
            accumulator: acc,
            innerPoly: innerPoly,
            pcsType: .ipa,
            ipaCheck: ipaCheck,
            transcript: &transcript
        )

        expectEqual(acc.foldCount, 2, "Full fold IPA: fold count = 2")
        expectEqual(acc.deferredIPAs.count, 1, "Full fold IPA: 1 deferred IPA")
        expectEqual(acc.deferredPairings.count, 0, "Full fold IPA: 0 deferred pairings")
    }

    // Test 8: Derive challenge is deterministic
    do {
        let poly = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let acc = RecursiveAccumulator.initial(poly: poly)

        let c1 = engine.deriveChallenge(accumulator: acc, domainSeparator: "test")
        let c2 = engine.deriveChallenge(accumulator: acc, domainSeparator: "test")
        expect(frEqual(c1, c2), "Derive challenge: deterministic")

        let c3 = engine.deriveChallenge(accumulator: acc, domainSeparator: "other")
        expect(!frEqual(c1, c3), "Derive challenge: different domain -> different challenge")
    }

    // Test 9: Polynomial evaluation
    do {
        // p(x) = 1 + 2x + 3x^2, p(2) = 1 + 4 + 12 = 17
        let poly = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let val = engine.evaluatePolynomial(poly, at: frFromInt(2))
        expect(frEqual(val, frFromInt(17)), "Poly eval: 1 + 2*2 + 3*4 = 17")

        // Empty polynomial evaluates to zero
        let empty = engine.evaluatePolynomial([], at: frFromInt(5))
        expect(frEqual(empty, Fr.zero), "Poly eval: empty -> 0")
    }

    // Test 10: verifyAllDeferred with empty checks
    do {
        let acc = RecursiveAccumulator.zero(size: 2)
        var transcript = RecursiveTranscript()
        let ok = engine.verifyAllDeferred(accumulator: acc, transcript: &transcript)
        expect(ok, "verifyAllDeferred: empty checks -> pass")
    }
}
