// Tests for GPUCQLookupEngine -- GPU-accelerated CQ (Cached Quotients) lookup argument

import Foundation
import zkMetal

public func runGPUCQLookupTests() {
    suite("GPUCQLookup")

    testGPUCQBasicProveVerify()
    testGPUCQRepeatedLookups()
    testGPUCQAllSameValue()
    testGPUCQSingleElementTable()
    testGPUCQAsymmetric()
    testGPUCQMultiplicitySums()
    testGPUCQTamperedMultiplicity()
    testGPUCQTamperedPhiCommitment()
    testGPUCQTamperedQuotientCommitment()
    testGPUCQWrongLookupCount()
    testGPUCQFiatShamirConsistency()
    testGPUCQPhiPolynomialEvaluation()
    testGPUCQLargerTable()
    testGPUCQSubsetLookup()
    testGPUCQBatchProve()
    testGPUCQBatchVerify()
    testGPUCQRoundTrip()
    testGPUCQTablePreprocessDeterminism()
    testGPUCQVanishingPoly()
    testGPUCQSparseMultiplicities()
    testGPUCQFullTableAccess()
    testGPUCQPairingVerification()
    testGPUCQMultipleBatchesAgainstSameTable()
    testGPUCQProofStructureFields()
}

// MARK: - Test SRS Helper

/// Build a small test SRS with a known secret for GPU CQ tests.
/// Returns (srs, secret) where secret is the Fr element.
private func buildTestSRS(size: Int) -> ([PointAffine], Fr) {
    let secretLimbs: [UInt32] = [0x1234_5678, 0x9ABC_DEF0, 0x1111_2222, 0x3333_4444,
                                  0x5555_6666, 0x7777_8888, 0x0000_0001, 0x0000_0000]
    let srs = KZGEngine.generateTestSRS(secret: secretLimbs, size: size,
                                         generator: bn254G1Generator())
    let secret = frFromLimbs(secretLimbs)
    return (srs, secret)
}

/// Build a power-of-2 table with distinct Fr values.
private func buildTable(size: Int) -> [Fr] {
    return (0..<size).map { frFromInt(UInt64($0 + 1)) }
}

/// Deterministic pseudo-random witness (all elements in table).
private func buildRandomWitness(table: [Fr], count: Int, seed: UInt64) -> [Fr] {
    let N = table.count
    var rng = seed
    var witness = [Fr]()
    witness.reserveCapacity(count)
    for _ in 0..<count {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        let idx = Int(rng >> 32) % N
        witness.append(table[idx])
    }
    return witness
}

// MARK: - Basic Prove/Verify

private func testGPUCQBasicProveVerify() {
    do {
        let (srs, secret) = buildTestSRS(size: 32)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 4)
        let preprocessed = try engine.preprocessTable(table: table)

        // Lookup all 4 elements
        let lookups: [Fr] = [frFromInt(2), frFromInt(1), frFromInt(4), frFromInt(3)]
        let proof = try engine.prove(lookups: lookups, table: preprocessed)
        let valid = engine.verify(proof: proof, table: preprocessed,
                                  numLookups: 4, srsSecret: secret)
        expect(valid, "Basic GPU CQ prove/verify (n=4, T=4)")
    } catch {
        expect(false, "Basic GPU CQ threw: \(error)")
    }
}

// MARK: - Repeated Lookups

private func testGPUCQRepeatedLookups() {
    do {
        let (srs, secret) = buildTestSRS(size: 32)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 4)
        let preprocessed = try engine.preprocessTable(table: table)

        // Repeated elements
        let lookups: [Fr] = [frFromInt(1), frFromInt(1), frFromInt(2), frFromInt(2),
                             frFromInt(3), frFromInt(3), frFromInt(4), frFromInt(4)]
        let proof = try engine.prove(lookups: lookups, table: preprocessed)
        let valid = engine.verify(proof: proof, table: preprocessed,
                                  numLookups: 8, srsSecret: secret)
        expect(valid, "Repeated lookups (n=8, T=4)")

        // Check multiplicities
        expect(frEqual(proof.multiplicities[0], frFromInt(2)), "m[0] == 2")
        expect(frEqual(proof.multiplicities[1], frFromInt(2)), "m[1] == 2")
        expect(frEqual(proof.multiplicities[2], frFromInt(2)), "m[2] == 2")
        expect(frEqual(proof.multiplicities[3], frFromInt(2)), "m[3] == 2")
    } catch {
        expect(false, "Repeated lookups threw: \(error)")
    }
}

// MARK: - All Same Value

private func testGPUCQAllSameValue() {
    do {
        let (srs, secret) = buildTestSRS(size: 32)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 4)
        let preprocessed = try engine.preprocessTable(table: table)

        // All lookups are the same value
        let lookups: [Fr] = [frFromInt(3), frFromInt(3), frFromInt(3), frFromInt(3)]
        let proof = try engine.prove(lookups: lookups, table: preprocessed)
        let valid = engine.verify(proof: proof, table: preprocessed,
                                  numLookups: 4, srsSecret: secret)
        expect(valid, "All same value (4x T[2])")

        // Only m[2] should be non-zero (T[2] = 3)
        expect(frEqual(proof.multiplicities[0], Fr.zero), "m[0] == 0")
        expect(frEqual(proof.multiplicities[1], Fr.zero), "m[1] == 0")
        expect(frEqual(proof.multiplicities[2], frFromInt(4)), "m[2] == 4")
        expect(frEqual(proof.multiplicities[3], Fr.zero), "m[3] == 0")
    } catch {
        expect(false, "All same value threw: \(error)")
    }
}

// MARK: - Single Element Table

private func testGPUCQSingleElementTable() {
    do {
        let (srs, secret) = buildTestSRS(size: 32)
        let engine = try GPUCQLookupEngine(srs: srs)

        // Minimal table: 1 element (power of 2 = 2^0 = 1)
        let table: [Fr] = [frFromInt(42)]
        let preprocessed = try engine.preprocessTable(table: table)

        let lookups: [Fr] = [frFromInt(42)]
        let proof = try engine.prove(lookups: lookups, table: preprocessed)
        let valid = engine.verify(proof: proof, table: preprocessed,
                                  numLookups: 1, srsSecret: secret)
        expect(valid, "Single element table (T=1, n=1)")
        expect(frEqual(proof.multiplicities[0], Fr.one), "Single multiplicity == 1")
    } catch {
        expect(false, "Single element table threw: \(error)")
    }
}

// MARK: - Asymmetric (small lookups into larger table)

private func testGPUCQAsymmetric() {
    do {
        let (srs, secret) = buildTestSRS(size: 64)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 16)
        let preprocessed = try engine.preprocessTable(table: table)

        // Only look up 3 elements from 16-element table
        let lookups: [Fr] = [frFromInt(5), frFromInt(10), frFromInt(15)]
        let proof = try engine.prove(lookups: lookups, table: preprocessed)
        let valid = engine.verify(proof: proof, table: preprocessed,
                                  numLookups: 3, srsSecret: secret)
        expect(valid, "Asymmetric (n=3, T=16)")

        // Check that only 3 entries have non-zero multiplicity
        var nonZeroCount = 0
        for i in 0..<16 {
            if !frEqual(proof.multiplicities[i], Fr.zero) {
                nonZeroCount += 1
            }
        }
        expectEqual(nonZeroCount, 3, "Exactly 3 non-zero multiplicities")
    } catch {
        expect(false, "Asymmetric threw: \(error)")
    }
}

// MARK: - Multiplicity Sums

private func testGPUCQMultiplicitySums() {
    do {
        let (srs, _) = buildTestSRS(size: 32)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 8)
        let preprocessed = try engine.preprocessTable(table: table)

        let lookups: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3),
                             frFromInt(1), frFromInt(2), frFromInt(1)]
        let proof = try engine.prove(lookups: lookups, table: preprocessed)

        // Sum of multiplicities should equal number of lookups
        expect(frEqual(proof.multiplicitySum, frFromInt(6)), "Multiplicity sum == 6")

        // Individual multiplicities
        expect(frEqual(proof.multiplicities[0], frFromInt(3)), "m[0] == 3 (value 1)")
        expect(frEqual(proof.multiplicities[1], frFromInt(2)), "m[1] == 2 (value 2)")
        expect(frEqual(proof.multiplicities[2], frFromInt(1)), "m[2] == 1 (value 3)")

        // Remaining should be zero
        for i in 3..<8 {
            expect(frEqual(proof.multiplicities[i], Fr.zero),
                   "m[\(i)] == 0 (unused)")
        }
    } catch {
        expect(false, "Multiplicity sums threw: \(error)")
    }
}

// MARK: - Tampered Multiplicity Rejected

private func testGPUCQTamperedMultiplicity() {
    do {
        let (srs, secret) = buildTestSRS(size: 32)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 4)
        let preprocessed = try engine.preprocessTable(table: table)

        let lookups: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let proof = try engine.prove(lookups: lookups, table: preprocessed)

        // Tamper: change a multiplicity
        var badMult = proof.multiplicities
        badMult[0] = frFromInt(5)  // was 1

        let tampered = GPUCQProof(
            phiCommitment: proof.phiCommitment,
            quotientCommitment: proof.quotientCommitment,
            multiplicities: badMult,
            multiplicitySum: proof.multiplicitySum,
            challengeZ: proof.challengeZ,
            phiOpening: proof.phiOpening,
            tOpening: proof.tOpening,
            phiCoeffs: proof.phiCoeffs
        )

        let rejected = !engine.verify(proof: tampered, table: preprocessed,
                                       numLookups: 4, srsSecret: secret)
        expect(rejected, "Tampered multiplicity rejected")
    } catch {
        expect(false, "Tampered multiplicity threw: \(error)")
    }
}

// MARK: - Tampered Phi Commitment Rejected

private func testGPUCQTamperedPhiCommitment() {
    do {
        let (srs, secret) = buildTestSRS(size: 32)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 4)
        let preprocessed = try engine.preprocessTable(table: table)

        let lookups: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let proof = try engine.prove(lookups: lookups, table: preprocessed)

        // Tamper: add G1 to phi commitment
        let badPhi = pointAdd(proof.phiCommitment,
                              pointFromAffine(bn254G1Generator()))

        let tampered = GPUCQProof(
            phiCommitment: badPhi,
            quotientCommitment: proof.quotientCommitment,
            multiplicities: proof.multiplicities,
            multiplicitySum: proof.multiplicitySum,
            challengeZ: proof.challengeZ,
            phiOpening: proof.phiOpening,
            tOpening: proof.tOpening,
            phiCoeffs: proof.phiCoeffs
        )

        let rejected = !engine.verify(proof: tampered, table: preprocessed,
                                       numLookups: 4, srsSecret: secret)
        expect(rejected, "Tampered phi commitment rejected")
    } catch {
        expect(false, "Tampered phi commitment threw: \(error)")
    }
}

// MARK: - Tampered Quotient Commitment Rejected

private func testGPUCQTamperedQuotientCommitment() {
    do {
        let (srs, secret) = buildTestSRS(size: 32)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 4)
        let preprocessed = try engine.preprocessTable(table: table)

        let lookups: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let proof = try engine.prove(lookups: lookups, table: preprocessed)

        // Tamper: shift quotient commitment
        let badQ = pointAdd(proof.quotientCommitment,
                            pointFromAffine(bn254G1Generator()))

        let tampered = GPUCQProof(
            phiCommitment: proof.phiCommitment,
            quotientCommitment: badQ,
            multiplicities: proof.multiplicities,
            multiplicitySum: proof.multiplicitySum,
            challengeZ: proof.challengeZ,
            phiOpening: proof.phiOpening,
            tOpening: proof.tOpening,
            phiCoeffs: proof.phiCoeffs
        )

        let rejected = !engine.verify(proof: tampered, table: preprocessed,
                                       numLookups: 4, srsSecret: secret)
        expect(rejected, "Tampered quotient commitment rejected")
    } catch {
        expect(false, "Tampered quotient commitment threw: \(error)")
    }
}

// MARK: - Wrong Lookup Count Rejected

private func testGPUCQWrongLookupCount() {
    do {
        let (srs, secret) = buildTestSRS(size: 32)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 4)
        let preprocessed = try engine.preprocessTable(table: table)

        let lookups: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let proof = try engine.prove(lookups: lookups, table: preprocessed)

        // Claim wrong number of lookups
        let rejected = !engine.verify(proof: proof, table: preprocessed,
                                       numLookups: 5, srsSecret: secret)
        expect(rejected, "Wrong lookup count rejected")

        let rejected2 = !engine.verify(proof: proof, table: preprocessed,
                                        numLookups: 3, srsSecret: secret)
        expect(rejected2, "Wrong lookup count (under) rejected")
    } catch {
        expect(false, "Wrong lookup count threw: \(error)")
    }
}

// MARK: - Fiat-Shamir Consistency

private func testGPUCQFiatShamirConsistency() {
    do {
        let (srs, secret) = buildTestSRS(size: 32)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 4)
        let preprocessed = try engine.preprocessTable(table: table)

        let lookups: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]

        // Run twice: challenge should be identical for same inputs
        let proof1 = try engine.prove(lookups: lookups, table: preprocessed)
        let proof2 = try engine.prove(lookups: lookups, table: preprocessed)

        expect(frEqual(proof1.challengeZ, proof2.challengeZ),
               "Fiat-Shamir challenge is deterministic")

        // Both should verify
        let v1 = engine.verify(proof: proof1, table: preprocessed,
                               numLookups: 4, srsSecret: secret)
        let v2 = engine.verify(proof: proof2, table: preprocessed,
                               numLookups: 4, srsSecret: secret)
        expect(v1, "First proof verifies")
        expect(v2, "Second proof verifies")
    } catch {
        expect(false, "Fiat-Shamir consistency threw: \(error)")
    }
}

// MARK: - Phi Polynomial Evaluation

private func testGPUCQPhiPolynomialEvaluation() {
    do {
        let (srs, _) = buildTestSRS(size: 32)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 4)
        let preprocessed = try engine.preprocessTable(table: table)

        let lookups: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let proof = try engine.prove(lookups: lookups, table: preprocessed)

        // Verify that phi(z) from the opening matches our direct evaluation
        let z = proof.challengeZ
        let phiAtZ = engine.evaluatePoly(proof.phiCoeffs, at: z)
        expect(frEqual(phiAtZ, proof.phiOpening.evaluation),
               "phi(z) matches opening evaluation")
    } catch {
        expect(false, "Phi polynomial evaluation threw: \(error)")
    }
}

// MARK: - Larger Table

private func testGPUCQLargerTable() {
    do {
        let (srs, secret) = buildTestSRS(size: 64)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 16)
        let preprocessed = try engine.preprocessTable(table: table)

        // 32 random lookups from 16-element table
        let witness = buildRandomWitness(table: table, count: 32, seed: 0xDEAD_BEEF)

        let proof = try engine.prove(lookups: witness, table: preprocessed)
        let valid = engine.verify(proof: proof, table: preprocessed,
                                  numLookups: 32, srsSecret: secret)
        expect(valid, "Larger table (n=32, T=16)")

        // Multiplicity sum should be 32
        expect(frEqual(proof.multiplicitySum, frFromInt(32)),
               "Multiplicity sum == 32")
    } catch {
        expect(false, "Larger table threw: \(error)")
    }
}

// MARK: - Subset Lookup (fewer lookups than table entries)

private func testGPUCQSubsetLookup() {
    do {
        let (srs, secret) = buildTestSRS(size: 64)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 16)
        let preprocessed = try engine.preprocessTable(table: table)

        // Only look up 2 of 16 entries
        let lookups: [Fr] = [frFromInt(8), frFromInt(16)]
        let proof = try engine.prove(lookups: lookups, table: preprocessed)
        let valid = engine.verify(proof: proof, table: preprocessed,
                                  numLookups: 2, srsSecret: secret)
        expect(valid, "Subset lookup (n=2, T=16)")
    } catch {
        expect(false, "Subset lookup threw: \(error)")
    }
}

// MARK: - Batch Prove

private func testGPUCQBatchProve() {
    do {
        let (srs, secret) = buildTestSRS(size: 32)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 4)
        let preprocessed = try engine.preprocessTable(table: table)

        let batch1: [Fr] = [frFromInt(1), frFromInt(2)]
        let batch2: [Fr] = [frFromInt(3), frFromInt(4)]
        let batch3: [Fr] = [frFromInt(1), frFromInt(1), frFromInt(1)]

        let proofs = try engine.proveBatch(
            lookupBatches: [batch1, batch2, batch3], table: preprocessed)

        expectEqual(proofs.count, 3, "3 batch proofs produced")

        // Verify each individually
        let v1 = engine.verify(proof: proofs[0], table: preprocessed,
                               numLookups: 2, srsSecret: secret)
        let v2 = engine.verify(proof: proofs[1], table: preprocessed,
                               numLookups: 2, srsSecret: secret)
        let v3 = engine.verify(proof: proofs[2], table: preprocessed,
                               numLookups: 3, srsSecret: secret)

        expect(v1, "Batch proof 1 valid")
        expect(v2, "Batch proof 2 valid")
        expect(v3, "Batch proof 3 valid")
    } catch {
        expect(false, "Batch prove threw: \(error)")
    }
}

// MARK: - Batch Verify

private func testGPUCQBatchVerify() {
    do {
        let (srs, secret) = buildTestSRS(size: 32)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 4)
        let preprocessed = try engine.preprocessTable(table: table)

        let batch1: [Fr] = [frFromInt(1), frFromInt(2)]
        let batch2: [Fr] = [frFromInt(3), frFromInt(4), frFromInt(1)]

        let proofs = try engine.proveBatch(
            lookupBatches: [batch1, batch2], table: preprocessed)

        let allValid = engine.verifyBatch(proofs: proofs, table: preprocessed,
                                           lookupCounts: [2, 3], srsSecret: secret)
        expect(allValid, "Batch verify all pass")

        // Wrong counts should fail
        let badCounts = engine.verifyBatch(proofs: proofs, table: preprocessed,
                                            lookupCounts: [3, 3], srsSecret: secret)
        expect(!badCounts, "Batch verify with wrong counts fails")
    } catch {
        expect(false, "Batch verify threw: \(error)")
    }
}

// MARK: - Round Trip

private func testGPUCQRoundTrip() {
    do {
        let (srs, secret) = buildTestSRS(size: 32)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 8)
        let preprocessed = try engine.preprocessTable(table: table)

        let lookups: [Fr] = [frFromInt(1), frFromInt(4), frFromInt(7), frFromInt(8),
                             frFromInt(2), frFromInt(5)]

        let (proof, valid) = try engine.proveAndVerify(
            lookups: lookups, table: preprocessed, srsSecret: secret)

        expect(valid, "Round-trip prove-and-verify (n=6, T=8)")
        expect(frEqual(proof.multiplicitySum, frFromInt(6)),
               "Round-trip multiplicity sum == 6")
    } catch {
        expect(false, "Round trip threw: \(error)")
    }
}

// MARK: - Preprocessing Determinism

private func testGPUCQTablePreprocessDeterminism() {
    do {
        let (srs, _) = buildTestSRS(size: 32)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 4)
        let p1 = try engine.preprocessTable(table: table)
        let p2 = try engine.preprocessTable(table: table)

        // Commitments should be identical
        let c1 = batchToAffine([p1.commitment])
        let c2 = batchToAffine([p2.commitment])
        expect(fpToInt(c1[0].x) == fpToInt(c2[0].x), "Table commitment x matches")
        expect(fpToInt(c1[0].y) == fpToInt(c2[0].y), "Table commitment y matches")

        // Table coefficients should match
        expectEqual(p1.tableCoeffs.count, p2.tableCoeffs.count, "Coeff count matches")
        for i in 0..<p1.tableCoeffs.count {
            expect(frEqual(p1.tableCoeffs[i], p2.tableCoeffs[i]),
                   "Table coeff[\(i)] matches")
        }

        // Cached quotient commitments should match
        expectEqual(p1.cachedQuotientCommitments.count,
                    p2.cachedQuotientCommitments.count,
                    "Quotient commitment count matches")
    } catch {
        expect(false, "Preprocessing determinism threw: \(error)")
    }
}

// MARK: - Vanishing Polynomial

private func testGPUCQVanishingPoly() {
    do {
        let (srs, _) = buildTestSRS(size: 32)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 4)
        let preprocessed = try engine.preprocessTable(table: table)

        // Z_T(omega^i) = omega^{iT} - 1 = 1 - 1 = 0 for all roots of unity
        // Verify that roots are correct: omega^T = 1
        let omega = preprocessed.roots[1]  // omega^1
        var omegaT = Fr.one
        for _ in 0..<4 {
            omegaT = frMul(omegaT, omega)
        }
        expect(frEqual(omegaT, Fr.one), "omega^T == 1 (root of unity)")

        // omega^0 should be 1
        expect(frEqual(preprocessed.roots[0], Fr.one), "omega^0 == 1")

        // Check logT
        expectEqual(preprocessed.logT, 2, "logT == 2 for T=4")
    } catch {
        expect(false, "Vanishing poly threw: \(error)")
    }
}

// MARK: - Sparse Multiplicities

private func testGPUCQSparseMultiplicities() {
    do {
        let (srs, secret) = buildTestSRS(size: 64)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 16)
        let preprocessed = try engine.preprocessTable(table: table)

        // Only access 1 out of 16 entries, but many times
        let lookups: [Fr] = [frFromInt(7), frFromInt(7), frFromInt(7),
                             frFromInt(7), frFromInt(7)]
        let proof = try engine.prove(lookups: lookups, table: preprocessed)
        let valid = engine.verify(proof: proof, table: preprocessed,
                                  numLookups: 5, srsSecret: secret)
        expect(valid, "Sparse multiplicities (1 of 16 entries, 5 lookups)")

        // Only 1 non-zero multiplicity
        var nonZero = 0
        for i in 0..<16 {
            if !frEqual(proof.multiplicities[i], Fr.zero) {
                nonZero += 1
            }
        }
        expectEqual(nonZero, 1, "Only 1 non-zero multiplicity in sparse case")
        expect(frEqual(proof.multiplicities[6], frFromInt(5)), "m[6] == 5 (value 7)")
    } catch {
        expect(false, "Sparse multiplicities threw: \(error)")
    }
}

// MARK: - Full Table Access

private func testGPUCQFullTableAccess() {
    do {
        let (srs, secret) = buildTestSRS(size: 64)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 8)
        let preprocessed = try engine.preprocessTable(table: table)

        // Access every table entry exactly once
        let lookups: [Fr] = (0..<8).map { frFromInt(UInt64($0 + 1)) }
        let proof = try engine.prove(lookups: lookups, table: preprocessed)
        let valid = engine.verify(proof: proof, table: preprocessed,
                                  numLookups: 8, srsSecret: secret)
        expect(valid, "Full table access (n=8, T=8, each once)")

        // Every multiplicity should be 1
        for i in 0..<8 {
            expect(frEqual(proof.multiplicities[i], Fr.one),
                   "m[\(i)] == 1 (full access)")
        }
    } catch {
        expect(false, "Full table access threw: \(error)")
    }
}

// MARK: - Pairing Verification

private func testGPUCQPairingVerification() {
    do {
        let secretLimbs: [UInt32] = [0x1234_5678, 0x9ABC_DEF0, 0x1111_2222, 0x3333_4444,
                                      0x5555_6666, 0x7777_8888, 0x0000_0001, 0x0000_0000]
        let secret = frFromLimbs(secretLimbs)
        let (srs, _) = buildTestSRS(size: 32)
        let engine = try GPUCQLookupEngine(srs: srs)

        // Build SRS G2 point: [tau] * G2
        let g2Gen = bn254G2Generator()
        let srsG2 = g2ToAffine(g2ScalarMul(g2FromAffine(g2Gen), frToInt(secret)))!

        let table = buildTable(size: 4)
        let preprocessed = try engine.preprocessTable(table: table)

        let lookups: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let proof = try engine.prove(lookups: lookups, table: preprocessed)

        // Verify with pairings (no SRS secret needed for this path)
        let valid = engine.verifyWithPairings(proof: proof, table: preprocessed,
                                               numLookups: 4, srsG2: srsG2)
        expect(valid, "Pairing-based verification")
    } catch {
        expect(false, "Pairing verification threw: \(error)")
    }
}

// MARK: - Multiple Batches Against Same Table

private func testGPUCQMultipleBatchesAgainstSameTable() {
    do {
        let (srs, secret) = buildTestSRS(size: 64)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 8)
        let preprocessed = try engine.preprocessTable(table: table)

        // 5 different proof batches, all against the same preprocessed table
        var allValid = true
        for seed: UInt64 in [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD, 0xEEEE] {
            let lookups = buildRandomWitness(table: table, count: 4, seed: seed)
            let proof = try engine.prove(lookups: lookups, table: preprocessed)
            let valid = engine.verify(proof: proof, table: preprocessed,
                                      numLookups: 4, srsSecret: secret)
            if !valid { allValid = false }
        }
        expect(allValid, "5 proof batches against same table all valid")
    } catch {
        expect(false, "Multiple batches threw: \(error)")
    }
}

// MARK: - Proof Structure Fields

private func testGPUCQProofStructureFields() {
    do {
        let (srs, _) = buildTestSRS(size: 32)
        let engine = try GPUCQLookupEngine(srs: srs)

        let table = buildTable(size: 4)
        let preprocessed = try engine.preprocessTable(table: table)

        let lookups: [Fr] = [frFromInt(1), frFromInt(2)]
        let proof = try engine.prove(lookups: lookups, table: preprocessed)

        // Verify proof structure fields are populated
        expectEqual(proof.multiplicities.count, 4, "Multiplicities length == T")
        expectEqual(proof.phiCoeffs.count, 4, "Phi coefficients length == T")
        expect(!frEqual(proof.challengeZ, Fr.zero), "Challenge z is non-zero")
        expect(frEqual(proof.multiplicitySum, frFromInt(2)), "Multiplicity sum == 2")

        // Phi commitment and quotient commitment should not be identity
        let phiAff = batchToAffine([proof.phiCommitment])
        let idAff = batchToAffine([pointIdentity()])
        let phiIsNotIdentity = !(fpToInt(phiAff[0].x) == fpToInt(idAff[0].x) &&
                                  fpToInt(phiAff[0].y) == fpToInt(idAff[0].y))
        expect(phiIsNotIdentity, "Phi commitment is not identity")

        // Table commitment fields
        expectEqual(preprocessed.table.count, 4, "Preprocessed table count == 4")
        expectEqual(preprocessed.tableCoeffs.count, 4, "Table coeffs count == 4")
        expectEqual(preprocessed.roots.count, 4, "Roots count == 4")
        expectEqual(preprocessed.logT, 2, "logT == 2")
        expectEqual(preprocessed.cachedQuotientCommitments.count, 4,
                    "Cached quotient commitments count == 4")
        expectEqual(preprocessed.cachedQuotientPolynomials.count, 4,
                    "Cached quotient polynomials count == 4")
    } catch {
        expect(false, "Proof structure fields threw: \(error)")
    }
}
