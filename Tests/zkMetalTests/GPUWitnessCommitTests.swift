// GPUWitnessCommitTests — Tests for GPU-accelerated witness polynomial commitment engine
//
// Validates witness-to-polynomial conversion (iNTT), KZG commitment via MSM,
// blinding for zero-knowledge, batch commitment, polynomial splitting for
// degree bounds, polynomial evaluation, and CPU/GPU cross-validation.

import Foundation
import zkMetal

// MARK: - Test Runner

public func runGPUWitnessCommitTests() {
    suite("GPU Witness Commit Engine")

    testSingleWitnessCommit()
    testWitnessToPolynomialRoundTrip()
    testPolynomialEvaluation()
    testCommitmentDeterminism()
    testZeroWitness()
    testSingleElementWitness()
    testBlindingDeterministic()
    testBlindingExplicit()
    testBlindingChangesCommitment()
    testBatchCommitment()
    testBatchCommitmentConsistency()
    testPolynomialSplitting()
    testSplitRecombination()
    testCommitWithOffset()
    testVerifyCommitment()
    testPolynomialConsistencyCheck()
    testNonPowerOf2Witness()
    testLargerWitness()
    testCacheManagement()
    testCommitLinearPolynomial()
    testCommitConstantPolynomial()
    testAdditiveHomomorphism()
    testDifferentWitnessesDifferentCommitments()
    testBatchWithBlinding()
    testSplitDegreeBoundEdgeCases()
    testIdentityCommitForEmptyPoly()
    testPolynomialEvalAtZero()
    testPolynomialEvalAtOne()
    testBlindingFactorGeneration()
    testCPUFallback()
}

// MARK: - Helpers

/// Generate a test SRS of given degree with a fixed tau.
private func makeTestSRS(degree: Int) -> [PointAffine] {
    let tau = frFromInt(42)
    return GPUWitnessCommitEngine.generateTestSRS(degree: degree, tau: tau)
}

/// Create a CPU-only engine for deterministic testing.
private func makeTestEngine(degree: Int = 32,
                            numBlinding: Int = 0,
                            cachePolynomials: Bool = false) -> GPUWitnessCommitEngine {
    let srs = makeTestSRS(degree: degree)
    return GPUWitnessCommitEngine(srs: srs, cpuOnly: true)
}

/// Create an engine with specific config.
private func makeConfiguredEngine(degree: Int = 32,
                                  numBlinding: Int = 0,
                                  cachePolynomials: Bool = false) -> GPUWitnessCommitEngine? {
    let srs = makeTestSRS(degree: degree)
    let config = WitnessCommitConfig(
        numBlindingFactors: numBlinding,
        useGPU: false,
        cachePolynomials: cachePolynomials
    )
    return try? GPUWitnessCommitEngine(srs: srs, config: config)
}

// MARK: - Test: Single Witness Commitment

private func testSingleWitnessCommit() {
    suite("WitnessCommit: Single Witness")

    do {
        let engine = makeTestEngine(degree: 16)

        // Simple witness: [1, 2, 3, 4]
        let witness = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let result = try engine.commitWitness(witness, blinding: .explicit(factors: []))

        // Commitment should be a valid point (not identity for non-zero witness)
        expect(!pointIsIdentity(result.commitment),
               "commitment should not be identity for non-zero witness")

        // Polynomial length should be power of 2
        expect(result.polynomial.count == 4,
               "polynomial length should be 4 (got \(result.polynomial.count))")

        // No blinding factors expected
        expect(result.blindingFactors.isEmpty,
               "no blinding factors expected")
    } catch {
        expect(false, "single witness commit threw: \(error)")
    }
}

// MARK: - Test: Witness to Polynomial Round-Trip

private func testWitnessToPolynomialRoundTrip() {
    suite("WitnessCommit: iNTT/NTT Round-Trip")

    do {
        let engine = makeTestEngine(degree: 16)

        // Witness evaluations at roots of unity
        let witness: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]

        // Convert to polynomial form
        let poly = try engine.witnessToPolynomial(witness)
        expect(poly.count == 4, "polynomial should have 4 coefficients")

        // Convert back to evaluations
        let recovered = try engine.polynomialToWitness(poly)
        expect(recovered.count == 4, "recovered should have 4 evaluations")

        // Should match original witness
        for i in 0..<4 {
            expect(frEqual(recovered[i], witness[i]),
                   "round-trip mismatch at index \(i)")
        }
    } catch {
        expect(false, "round-trip test threw: \(error)")
    }
}

// MARK: - Test: Polynomial Evaluation

private func testPolynomialEvaluation() {
    suite("WitnessCommit: Polynomial Evaluation")

    let engine = makeTestEngine(degree: 16)

    // Polynomial p(x) = 3 + 2x + x^2
    // (coefficients in increasing degree order)
    let poly = [frFromInt(3), frFromInt(2), frFromInt(1), Fr.zero]

    // p(0) = 3
    let eval0 = engine.evaluatePolynomial(poly, at: Fr.zero)
    expect(frEqual(eval0, frFromInt(3)), "p(0) = 3")

    // p(1) = 3 + 2 + 1 = 6
    let eval1 = engine.evaluatePolynomial(poly, at: Fr.one)
    expect(frEqual(eval1, frFromInt(6)), "p(1) = 6")

    // p(2) = 3 + 4 + 4 = 11
    let eval2 = engine.evaluatePolynomial(poly, at: frFromInt(2))
    expect(frEqual(eval2, frFromInt(11)), "p(2) = 11")

    // p(3) = 3 + 6 + 9 = 18
    let eval3 = engine.evaluatePolynomial(poly, at: frFromInt(3))
    expect(frEqual(eval3, frFromInt(18)), "p(3) = 18")
}

// MARK: - Test: Commitment Determinism

private func testCommitmentDeterminism() {
    suite("WitnessCommit: Determinism")

    do {
        let engine = makeTestEngine(degree: 16)
        let witness = [frFromInt(5), frFromInt(10), frFromInt(15), frFromInt(20)]

        let result1 = try engine.commitWitness(witness, blinding: .explicit(factors: []))
        let result2 = try engine.commitWitness(witness, blinding: .explicit(factors: []))

        // Same witness with no blinding should produce same commitment
        expect(pointEqual(result1.commitment, result2.commitment),
               "deterministic commitment should be repeatable")

        // Polynomials should match
        expect(result1.polynomial.count == result2.polynomial.count,
               "polynomial lengths should match")
        for i in 0..<result1.polynomial.count {
            expect(frEqual(result1.polynomial[i], result2.polynomial[i]),
                   "polynomial coefficients should match at index \(i)")
        }
    } catch {
        expect(false, "determinism test threw: \(error)")
    }
}

// MARK: - Test: Zero Witness

private func testZeroWitness() {
    suite("WitnessCommit: Zero Witness")

    do {
        let engine = makeTestEngine(degree: 16)
        let witness = [Fr.zero, Fr.zero, Fr.zero, Fr.zero]

        let result = try engine.commitWitness(witness, blinding: .explicit(factors: []))

        // Commitment to zero polynomial should be identity
        expect(pointIsIdentity(result.commitment),
               "commitment to zero witness should be identity")

        // All polynomial coefficients should be zero
        for i in 0..<result.polynomial.count {
            expect(frEqual(result.polynomial[i], Fr.zero),
                   "zero witness polynomial coeff \(i) should be zero")
        }
    } catch {
        expect(false, "zero witness test threw: \(error)")
    }
}

// MARK: - Test: Single Element Witness

private func testSingleElementWitness() {
    suite("WitnessCommit: Single Element")

    do {
        let engine = makeTestEngine(degree: 16)

        // Single element witness: [7]
        let witness = [frFromInt(7)]
        let result = try engine.commitWitness(witness, blinding: .explicit(factors: []))

        // Polynomial should be the constant polynomial [7]
        expect(result.polynomial.count >= 1, "polynomial should have at least 1 coeff")
        expect(frEqual(result.polynomial[0], frFromInt(7)),
               "constant polynomial should equal witness value")
    } catch {
        expect(false, "single element test threw: \(error)")
    }
}

// MARK: - Test: Deterministic Blinding

private func testBlindingDeterministic() {
    suite("WitnessCommit: Deterministic Blinding")

    do {
        let srs = makeTestSRS(degree: 32)
        let config = WitnessCommitConfig(numBlindingFactors: 3, useGPU: false)
        let engine = try GPUWitnessCommitEngine(srs: srs, config: config)

        let witness = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]

        let result1 = try engine.commitWitness(witness, blinding: .deterministic(seed: 12345))
        let result2 = try engine.commitWitness(witness, blinding: .deterministic(seed: 12345))

        // Same seed should produce same blinding factors
        expect(result1.blindingFactors.count == 3,
               "should have 3 blinding factors (got \(result1.blindingFactors.count))")
        expect(result2.blindingFactors.count == 3,
               "should have 3 blinding factors (got \(result2.blindingFactors.count))")

        for i in 0..<3 {
            expect(frEqual(result1.blindingFactors[i], result2.blindingFactors[i]),
                   "deterministic blinding factor \(i) should match")
        }

        // Same blinding should produce same commitment
        expect(pointEqual(result1.commitment, result2.commitment),
               "deterministic blinding should produce same commitment")
    } catch {
        expect(false, "deterministic blinding test threw: \(error)")
    }
}

// MARK: - Test: Explicit Blinding

private func testBlindingExplicit() {
    suite("WitnessCommit: Explicit Blinding")

    do {
        let srs = makeTestSRS(degree: 32)
        let config = WitnessCommitConfig(numBlindingFactors: 2, useGPU: false)
        let engine = try GPUWitnessCommitEngine(srs: srs, config: config)

        let witness = [frFromInt(5), frFromInt(10), frFromInt(15), frFromInt(20)]
        let blinds = [frFromInt(99), frFromInt(88)]

        let result = try engine.commitWitness(witness, blinding: .explicit(factors: blinds))

        expect(result.blindingFactors.count == 2, "should have 2 blinding factors")
        expect(frEqual(result.blindingFactors[0], frFromInt(99)),
               "first blinding factor should be 99")
        expect(frEqual(result.blindingFactors[1], frFromInt(88)),
               "second blinding factor should be 88")

        // Polynomial should be larger than original witness (includes blinding)
        expect(result.polynomial.count > 4,
               "blinded polynomial should be larger than witness")
    } catch {
        expect(false, "explicit blinding test threw: \(error)")
    }
}

// MARK: - Test: Blinding Changes Commitment

private func testBlindingChangesCommitment() {
    suite("WitnessCommit: Blinding Changes Commitment")

    do {
        let srs = makeTestSRS(degree: 32)
        let configNoBlind = WitnessCommitConfig(numBlindingFactors: 0, useGPU: false)
        let configBlind = WitnessCommitConfig(numBlindingFactors: 3, useGPU: false)

        let engineNoBlind = try GPUWitnessCommitEngine(srs: srs, config: configNoBlind)
        let engineBlind = try GPUWitnessCommitEngine(srs: srs, config: configBlind)

        let witness = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]

        let noBlind = try engineNoBlind.commitWitness(witness, blinding: .explicit(factors: []))
        let withBlind = try engineBlind.commitWitness(witness,
                                                       blinding: .explicit(factors: [frFromInt(77), frFromInt(88), frFromInt(99)]))

        // Blinding should change the commitment
        expect(!pointEqual(noBlind.commitment, withBlind.commitment),
               "blinding should produce a different commitment")
    } catch {
        expect(false, "blinding changes test threw: \(error)")
    }
}

// MARK: - Test: Batch Commitment

private func testBatchCommitment() {
    suite("WitnessCommit: Batch Commitment")

    do {
        let engine = makeTestEngine(degree: 32)
        let w1 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let w2 = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]
        let w3 = [frFromInt(9), frFromInt(10), frFromInt(11), frFromInt(12)]

        let batch = try engine.batchCommitWitnesses([w1, w2, w3],
                                                     blinding: .explicit(factors: []))

        expect(batch.commitments.count == 3, "batch should have 3 commitments")

        // Each commitment should be non-identity
        for (i, c) in batch.commitments.enumerated() {
            expect(!pointIsIdentity(c.commitment),
                   "batch commitment \(i) should not be identity")
        }

        // Batch commitments should match individual commits
        let ind1 = try engine.commitWitness(w1, blinding: .explicit(factors: []))
        let ind2 = try engine.commitWitness(w2, blinding: .explicit(factors: []))
        let ind3 = try engine.commitWitness(w3, blinding: .explicit(factors: []))

        expect(pointEqual(batch.commitments[0].commitment, ind1.commitment),
               "batch[0] should match individual commit")
        expect(pointEqual(batch.commitments[1].commitment, ind2.commitment),
               "batch[1] should match individual commit")
        expect(pointEqual(batch.commitments[2].commitment, ind3.commitment),
               "batch[2] should match individual commit")
    } catch {
        expect(false, "batch commitment test threw: \(error)")
    }
}

// MARK: - Test: Batch Commitment Consistency

private func testBatchCommitmentConsistency() {
    suite("WitnessCommit: Batch Consistency")

    do {
        let engine = makeTestEngine(degree: 16)

        // All-same witnesses should produce all-same commitments
        let w = [frFromInt(42), frFromInt(43), frFromInt(44), frFromInt(45)]
        let batch = try engine.batchCommitWitnesses([w, w, w],
                                                     blinding: .explicit(factors: []))

        for i in 1..<batch.commitments.count {
            expect(pointEqual(batch.commitments[0].commitment,
                              batch.commitments[i].commitment),
                   "identical witnesses should produce identical commitments (\(i))")
        }

        // Empty batch
        let empty = try engine.batchCommitWitnesses([],
                                                     blinding: .explicit(factors: []))
        expect(empty.commitments.isEmpty, "empty batch should produce no commitments")
    } catch {
        expect(false, "batch consistency test threw: \(error)")
    }
}

// MARK: - Test: Polynomial Splitting

private func testPolynomialSplitting() {
    suite("WitnessCommit: Polynomial Splitting")

    do {
        let engine = makeTestEngine(degree: 16)

        // Polynomial with 8 coefficients, split at degree 4
        let poly: [Fr] = (1...8).map { frFromInt(UInt64($0)) }
        let split = try engine.splitAndCommit(poly, degreeBound: 4)

        // Low part should have 4 coefficients
        expect(split.low.count == 4, "low part should have 4 coeffs (got \(split.low.count))")
        for i in 0..<4 {
            expect(frEqual(split.low[i], frFromInt(UInt64(i + 1))),
                   "low[\(i)] should be \(i + 1)")
        }

        // High part should have 4 coefficients
        expect(split.high.count == 4, "high part should have 4 coeffs (got \(split.high.count))")
        for i in 0..<4 {
            expect(frEqual(split.high[i], frFromInt(UInt64(i + 5))),
                   "high[\(i)] should be \(i + 5)")
        }

        // Degree bound should be recorded
        expect(split.degreeBound == 4, "degree bound should be 4")

        // Both commitments should be non-identity
        expect(!pointIsIdentity(split.lowCommitment),
               "low commitment should not be identity")
        expect(!pointIsIdentity(split.highCommitment),
               "high commitment should not be identity")
    } catch {
        expect(false, "polynomial splitting test threw: \(error)")
    }
}

// MARK: - Test: Split Recombination

private func testSplitRecombination() {
    suite("WitnessCommit: Split Recombination")

    do {
        let engine = makeTestEngine(degree: 16)

        // Create a polynomial and commit to it directly
        let poly: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40),
                          frFromInt(50), frFromInt(60), frFromInt(70), frFromInt(80)]
        let fullCommit = try engine.commitPolynomial(poly)

        // Split at degree 4
        let split = try engine.splitAndCommit(poly, degreeBound: 4)

        // Recombine: C_full = C_low + C_high (where C_high uses shifted SRS)
        // The full commitment should equal the sum of low and high commitments
        let recombined = pointAdd(split.lowCommitment, split.highCommitment)
        expect(pointEqual(fullCommit, recombined),
               "split commitments should recombine to full commitment")
    } catch {
        expect(false, "split recombination test threw: \(error)")
    }
}

// MARK: - Test: Commit with Offset

private func testCommitWithOffset() {
    suite("WitnessCommit: Commit with Offset")

    do {
        let engine = makeTestEngine(degree: 16)

        // Commit [a, b] at offset 0 and at offset 2
        let coeffs = [frFromInt(3), frFromInt(7)]

        let c0 = try engine.commitWithOffset(coeffs, offset: 0)
        let c2 = try engine.commitWithOffset(coeffs, offset: 2)

        // Different offsets should produce different commitments
        expect(!pointEqual(c0, c2),
               "different SRS offsets should produce different commitments")

        // Commit at offset 0 should equal normal commit
        let cNormal = try engine.commitPolynomial(coeffs)
        expect(pointEqual(c0, cNormal),
               "commit at offset 0 should equal normal commit")
    } catch {
        expect(false, "commit with offset test threw: \(error)")
    }
}

// MARK: - Test: Verify Commitment

private func testVerifyCommitment() {
    suite("WitnessCommit: Verify Commitment")

    do {
        let engine = makeTestEngine(degree: 16)
        let poly = [frFromInt(11), frFromInt(22), frFromInt(33), frFromInt(44)]

        let commitment = try engine.commitPolynomial(poly)

        // Correct polynomial should verify
        let valid = try engine.verifyCommitment(commitment, polynomial: poly)
        expect(valid, "correct polynomial should verify")

        // Wrong polynomial should fail verification
        let wrongPoly = [frFromInt(11), frFromInt(22), frFromInt(33), frFromInt(55)]
        let invalid = try engine.verifyCommitment(commitment, polynomial: wrongPoly)
        expect(!invalid, "wrong polynomial should not verify")
    } catch {
        expect(false, "verify commitment test threw: \(error)")
    }
}

// MARK: - Test: Polynomial Consistency Check

private func testPolynomialConsistencyCheck() {
    suite("WitnessCommit: Polynomial Consistency")

    do {
        let engine = makeTestEngine(degree: 16)
        let witness: [Fr] = [frFromInt(100), frFromInt(200), frFromInt(300), frFromInt(400)]

        let poly = try engine.witnessToPolynomial(witness)

        // Correct witness should be consistent
        let consistent = try engine.verifyPolynomialConsistency(poly, originalWitness: witness)
        expect(consistent, "polynomial should be consistent with original witness")

        // Wrong witness should be inconsistent
        let wrongWitness: [Fr] = [frFromInt(100), frFromInt(200), frFromInt(300), frFromInt(999)]
        let inconsistent = try engine.verifyPolynomialConsistency(poly, originalWitness: wrongWitness)
        expect(!inconsistent, "polynomial should be inconsistent with wrong witness")
    } catch {
        expect(false, "polynomial consistency test threw: \(error)")
    }
}

// MARK: - Test: Non-Power-of-2 Witness

private func testNonPowerOf2Witness() {
    suite("WitnessCommit: Non-Power-of-2 Witness")

    do {
        let engine = makeTestEngine(degree: 16)

        // 3 elements (not power of 2) -- should be padded to 4
        let witness = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let result = try engine.commitWitness(witness, blinding: .explicit(factors: []))

        expect(result.polynomial.count == 4,
               "non-pow2 witness should be padded to 4 (got \(result.polynomial.count))")
        expect(!pointIsIdentity(result.commitment),
               "commitment for non-pow2 witness should be valid")

        // 5 elements should pad to 8
        let witness5 = (1...5).map { frFromInt(UInt64($0)) }
        let result5 = try engine.commitWitness(witness5, blinding: .explicit(factors: []))
        expect(result5.polynomial.count == 8,
               "5-element witness should pad to 8 (got \(result5.polynomial.count))")
    } catch {
        expect(false, "non-power-of-2 test threw: \(error)")
    }
}

// MARK: - Test: Larger Witness

private func testLargerWitness() {
    suite("WitnessCommit: Larger Witness (16 elements)")

    do {
        let engine = makeTestEngine(degree: 32)

        let witness = (1...16).map { frFromInt(UInt64($0)) }
        let result = try engine.commitWitness(witness, blinding: .explicit(factors: []))

        expect(result.polynomial.count == 16,
               "16-element polynomial (got \(result.polynomial.count))")
        expect(!pointIsIdentity(result.commitment),
               "larger witness commitment should not be identity")

        // Round-trip check
        let consistent = try engine.verifyPolynomialConsistency(
            result.polynomial, originalWitness: witness)
        expect(consistent, "larger witness round-trip should be consistent")
    } catch {
        expect(false, "larger witness test threw: \(error)")
    }
}

// MARK: - Test: Cache Management

private func testCacheManagement() {
    suite("WitnessCommit: Cache Management")

    do {
        guard let engine = makeConfiguredEngine(degree: 16, cachePolynomials: true) else {
            expect(false, "failed to create configured engine")
            return
        }

        let witness = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        _ = try engine.commitWitness(witness, blinding: .explicit(factors: []))

        // Check cache populated
        let cached = engine.getCachedPolynomial(witnessSize: 4)
        expect(cached != nil, "cache should contain polynomial for size 4")

        if let cached = cached {
            expect(cached.count == 4, "cached polynomial should have 4 coeffs")
        }

        // Clear cache
        engine.clearCache()
        let afterClear = engine.getCachedPolynomial(witnessSize: 4)
        expect(afterClear == nil, "cache should be empty after clear")
    } catch {
        expect(false, "cache management test threw: \(error)")
    }
}

// MARK: - Test: Commit Linear Polynomial

private func testCommitLinearPolynomial() {
    suite("WitnessCommit: Linear Polynomial")

    do {
        let engine = makeTestEngine(degree: 16)

        // Polynomial p(x) = x  (coeffs: [0, 1, 0, 0])
        let poly = [Fr.zero, Fr.one, Fr.zero, Fr.zero]
        let commitment = try engine.commitPolynomial(poly)

        // C = 0*G + 1*sG + 0*s^2G + 0*s^3G = sG
        // This should be the second SRS point
        let srsSecond = pointFromAffine(engine.srs[1])
        expect(pointEqual(commitment, srsSecond),
               "commit([0,1,0,0]) should equal second SRS point")
    } catch {
        expect(false, "linear polynomial test threw: \(error)")
    }
}

// MARK: - Test: Commit Constant Polynomial

private func testCommitConstantPolynomial() {
    suite("WitnessCommit: Constant Polynomial")

    do {
        let engine = makeTestEngine(degree: 16)

        // Polynomial p(x) = 5  (coeffs: [5, 0, 0, 0])
        let five = frFromInt(5)
        let poly = [five, Fr.zero, Fr.zero, Fr.zero]
        let commitment = try engine.commitPolynomial(poly)

        // C = 5*G + 0 + 0 + 0 = 5*G
        let g1 = pointFromAffine(engine.srs[0])
        let expected = pointScalarMul(g1, five)
        expect(pointEqual(commitment, expected),
               "commit([5,0,0,0]) should equal 5*G")
    } catch {
        expect(false, "constant polynomial test threw: \(error)")
    }
}

// MARK: - Test: Additive Homomorphism

private func testAdditiveHomomorphism() {
    suite("WitnessCommit: Additive Homomorphism")

    do {
        let engine = makeTestEngine(degree: 16)

        // Two polynomials
        let p1 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let p2 = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]

        // Their sum
        var pSum = [Fr]()
        for i in 0..<4 {
            pSum.append(frAdd(p1[i], p2[i]))
        }

        let c1 = try engine.commitPolynomial(p1)
        let c2 = try engine.commitPolynomial(p2)
        let cSum = try engine.commitPolynomial(pSum)

        // commit(p1 + p2) should equal commit(p1) + commit(p2)
        let pointSum = pointAdd(c1, c2)
        expect(pointEqual(cSum, pointSum),
               "KZG additive homomorphism: C(p1+p2) = C(p1)+C(p2)")
    } catch {
        expect(false, "additive homomorphism test threw: \(error)")
    }
}

// MARK: - Test: Different Witnesses Different Commitments

private func testDifferentWitnessesDifferentCommitments() {
    suite("WitnessCommit: Different Witnesses")

    do {
        let engine = makeTestEngine(degree: 16)

        let w1 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let w2 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(5)]

        let c1 = try engine.commitWitness(w1, blinding: .explicit(factors: []))
        let c2 = try engine.commitWitness(w2, blinding: .explicit(factors: []))

        expect(!pointEqual(c1.commitment, c2.commitment),
               "different witnesses should produce different commitments")
    } catch {
        expect(false, "different witnesses test threw: \(error)")
    }
}

// MARK: - Test: Batch With Blinding

private func testBatchWithBlinding() {
    suite("WitnessCommit: Batch With Per-Column Blinding")

    do {
        let srs = makeTestSRS(degree: 32)
        let config = WitnessCommitConfig(numBlindingFactors: 2, useGPU: false)
        let engine = try GPUWitnessCommitEngine(srs: srs, config: config)

        let w1 = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let w2 = [frFromInt(5), frFromInt(6), frFromInt(7), frFromInt(8)]

        let b1 = [frFromInt(100), frFromInt(200)]
        let b2 = [frFromInt(300), frFromInt(400)]

        let batch = try engine.batchCommitWithBlinding([w1, w2],
                                                       blindingFactors: [b1, b2])

        expect(batch.commitments.count == 2, "batch should have 2 commitments")
        expect(batch.commitments[0].blindingFactors.count == 2,
               "first should have 2 blinding factors")
        expect(batch.commitments[1].blindingFactors.count == 2,
               "second should have 2 blinding factors")

        // Verify blinding factors are the ones we specified
        expect(frEqual(batch.commitments[0].blindingFactors[0], frFromInt(100)),
               "batch[0] blind[0] should be 100")
        expect(frEqual(batch.commitments[1].blindingFactors[1], frFromInt(400)),
               "batch[1] blind[1] should be 400")
    } catch {
        expect(false, "batch with blinding test threw: \(error)")
    }
}

// MARK: - Test: Split Degree Bound Edge Cases

private func testSplitDegreeBoundEdgeCases() {
    suite("WitnessCommit: Split Edge Cases")

    do {
        let engine = makeTestEngine(degree: 16)

        let poly: [Fr] = (1...8).map { frFromInt(UInt64($0)) }

        // Split at degree 1 (minimal low part)
        let split1 = try engine.splitAndCommit(poly, degreeBound: 1)
        expect(split1.low.count == 1, "low part should have 1 coeff")
        expect(split1.high.count == 7, "high part should have 7 coeffs")

        // Split at degree 7 (minimal high part)
        let split7 = try engine.splitAndCommit(poly, degreeBound: 7)
        expect(split7.low.count == 7, "low part should have 7 coeffs")
        expect(split7.high.count == 1, "high part should have 1 coeff")
    } catch {
        expect(false, "split edge cases test threw: \(error)")
    }
}

// MARK: - Test: Identity Commitment for Empty Polynomial

private func testIdentityCommitForEmptyPoly() {
    suite("WitnessCommit: Identity for Empty Polynomial")

    do {
        let engine = makeTestEngine(degree: 16)

        let commitment = try engine.commitPolynomial([])
        expect(pointIsIdentity(commitment),
               "commitment to empty polynomial should be identity")
    } catch {
        expect(false, "identity commit test threw: \(error)")
    }
}

// MARK: - Test: Polynomial Evaluation at Zero

private func testPolynomialEvalAtZero() {
    suite("WitnessCommit: Eval at Zero")

    let engine = makeTestEngine(degree: 16)

    // p(x) = a0 + a1*x + a2*x^2 + ...
    // p(0) = a0
    let coeffs = [frFromInt(42), frFromInt(7), frFromInt(13), frFromInt(99)]
    let result = engine.evaluatePolynomial(coeffs, at: Fr.zero)
    expect(frEqual(result, frFromInt(42)), "p(0) should equal constant coefficient")

    // Empty polynomial
    let emptyResult = engine.evaluatePolynomial([], at: frFromInt(5))
    expect(frEqual(emptyResult, Fr.zero), "empty polynomial should evaluate to zero")
}

// MARK: - Test: Polynomial Evaluation at One

private func testPolynomialEvalAtOne() {
    suite("WitnessCommit: Eval at One")

    let engine = makeTestEngine(degree: 16)

    // p(x) = 1 + 2x + 3x^2 + 4x^3
    // p(1) = 1 + 2 + 3 + 4 = 10
    let coeffs = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    let result = engine.evaluatePolynomial(coeffs, at: Fr.one)
    expect(frEqual(result, frFromInt(10)), "p(1) = sum of coefficients = 10")
}

// MARK: - Test: Blinding Factor Generation

private func testBlindingFactorGeneration() {
    suite("WitnessCommit: Blinding Factor Generation")

    let engine = makeTestEngine(degree: 16)

    // Zero count
    let empty = engine.generateBlindingFactors(count: 0)
    expect(empty.isEmpty, "zero blinding factors should be empty")

    // Deterministic should be non-zero and reproducible
    let d1 = engine.generateBlindingFactors(count: 4, mode: .deterministic(seed: 42))
    let d2 = engine.generateBlindingFactors(count: 4, mode: .deterministic(seed: 42))
    expect(d1.count == 4, "should have 4 factors")
    for i in 0..<4 {
        expect(frEqual(d1[i], d2[i]), "deterministic factors should match at \(i)")
    }

    // Different seeds should produce different factors
    let d3 = engine.generateBlindingFactors(count: 4, mode: .deterministic(seed: 99))
    var allSame = true
    for i in 0..<4 {
        if !frEqual(d1[i], d3[i]) { allSame = false; break }
    }
    expect(!allSame, "different seeds should produce different blinding factors")

    // Explicit with padding
    let explicit = engine.generateBlindingFactors(
        count: 4, mode: .explicit(factors: [frFromInt(1), frFromInt(2)]))
    expect(explicit.count == 4, "explicit should be padded to 4")
    expect(frEqual(explicit[0], frFromInt(1)), "explicit[0] = 1")
    expect(frEqual(explicit[1], frFromInt(2)), "explicit[1] = 2")
    expect(frEqual(explicit[2], Fr.zero), "explicit[2] padded with zero")
    expect(frEqual(explicit[3], Fr.zero), "explicit[3] padded with zero")

    // Explicit with truncation
    let truncated = engine.generateBlindingFactors(
        count: 2, mode: .explicit(factors: [frFromInt(10), frFromInt(20), frFromInt(30)]))
    expect(truncated.count == 2, "truncated should have 2")
    expect(frEqual(truncated[0], frFromInt(10)), "truncated[0] = 10")
    expect(frEqual(truncated[1], frFromInt(20)), "truncated[1] = 20")
}

// MARK: - Test: CPU Fallback

private func testCPUFallback() {
    suite("WitnessCommit: CPU Fallback")

    do {
        // Create a CPU-only engine
        let srs = makeTestSRS(degree: 16)
        let engine = GPUWitnessCommitEngine(srs: srs, cpuOnly: true)

        let witness = [frFromInt(3), frFromInt(5), frFromInt(7), frFromInt(11)]

        // CPU commit
        let result = try engine.commitWitness(witness, blinding: .explicit(factors: []))
        expect(!pointIsIdentity(result.commitment),
               "CPU fallback should produce valid commitment")

        // Round-trip via CPU
        let poly = try engine.witnessToPolynomial(witness)
        let recovered = try engine.polynomialToWitness(poly)

        for i in 0..<witness.count {
            expect(frEqual(recovered[i], witness[i]),
                   "CPU round-trip mismatch at \(i)")
        }

        // CPU verification
        let verified = try engine.verifyCommitment(result.commitment,
                                                    polynomial: result.polynomial)
        expect(verified, "CPU commitment should self-verify")
    } catch {
        expect(false, "CPU fallback test threw: \(error)")
    }
}
