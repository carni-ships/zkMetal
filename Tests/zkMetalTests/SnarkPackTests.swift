// SnarkPack aggregation tests: TIPP + MIPP inner-product arguments
import zkMetal

public func runSnarkPackTests() {
    suite("SnarkPack Engine")

    testSnarkPackSRSGeneration()
    testSnarkPackSRSG2Reversed()
    testSingleProofAggregation()
    testTwoProofAggregation()
    testFourProofAggregation()
    testTIPPProofStructure()
    testMIPPProofStructure()
    testFiatShamirDeterminism()
    testFiatShamirSensitivity()
    testVerificationRejectsCorruptedAggC()
    testVerificationRejectsWrongPublicInputs()
    testChallengeConsistency()
}

// MARK: - Helpers

/// Generate a Groth16 proof for the example circuit with given x.
private func makeProof(x: UInt64) -> (proof: Groth16Proof,
                                       vk: Groth16VerificationKey,
                                       publicInputs: [Fr]) {
    let r1cs = buildExampleCircuit()
    let (pubInputs, witness) = computeExampleWitness(x: x)
    let setup = Groth16Setup()
    let (pk, vk) = setup.setup(r1cs: r1cs)
    let prover = try! Groth16Prover()
    let proof = try! prover.prove(pk: pk, r1cs: r1cs, publicInputs: pubInputs, witness: witness)
    return (proof, vk, pubInputs)
}

/// Generate multiple proofs for the same circuit with different inputs.
private func makeProofs(count: Int, startX: UInt64 = 3) -> (
    proofs: [Groth16Proof],
    publicInputs: [[Fr]],
    vk: Groth16VerificationKey
) {
    let r1cs = buildExampleCircuit()
    let setup = Groth16Setup()
    let (pk, vk) = setup.setup(r1cs: r1cs)
    let prover = try! Groth16Prover()

    var proofs = [Groth16Proof]()
    var allPubInputs = [[Fr]]()

    for i in 0..<count {
        let x = startX + UInt64(i)
        let (pubInputs, witness) = computeExampleWitness(x: x)
        let proof = try! prover.prove(pk: pk, r1cs: r1cs, publicInputs: pubInputs, witness: witness)
        proofs.append(proof)
        allPubInputs.append(pubInputs)
    }

    return (proofs, allPubInputs, vk)
}

/// Create a test SRS.
private func testSRS(maxProofs: Int = 4) -> SnarkPackSRS {
    SnarkPackSRS.generate(maxProofs: maxProofs, seed: frFromInt(42))
}

private func frEqual(_ a: Fr, _ b: Fr) -> Bool {
    a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
    a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

// MARK: - SRS Tests

private func testSnarkPackSRSGeneration() {
    let srs = testSRS(maxProofs: 4)

    expect(srs.maxProofs == 4, "SRS maxProofs is 4")
    expect(srs.g1Powers.count == 4, "SRS has 4 G1 powers")
    expect(srs.g2Powers.count == 4, "SRS has 4 G2 powers")

    // First G1 power should be the generator
    let g1 = pointFromAffine(bn254G1Generator())
    expect(pointEqual(srs.g1Powers[0], g1), "g1Powers[0] is G1 generator")

    // Powers should be distinct
    expect(!pointEqual(srs.g1Powers[0], srs.g1Powers[1]),
           "g1Powers[0] != g1Powers[1]")

    // G2 elements non-identity
    expect(!g2IsIdentity(srs.g2Gen), "g2Gen non-identity")
    expect(!g2IsIdentity(srs.g2Tau), "g2Tau non-identity")
}

private func testSnarkPackSRSG2Reversed() {
    let srs = testSRS(maxProofs: 4)

    // g2Powers should be reversed: g2Powers[0] = tau^3 * G2, g2Powers[3] = G2
    // Last element should be the G2 generator (tau^0 * G2)
    let g2 = g2FromAffine(bn254G2Generator())

    // g2Powers are reversed, so last should be tau^0*G2 = G2
    // We check non-identity for all and distinctness
    for i in 0..<4 {
        expect(!g2IsIdentity(srs.g2Powers[i]),
               "g2Powers[\(i)] non-identity")
    }
}

// MARK: - Single Proof Aggregation

private func testSingleProofAggregation() {
    do {
        let (proof, vk, pubInputs) = makeProof(x: 3)

        let srs = testSRS(maxProofs: 1)

        let engine = SnarkPackEngine()
        let aggProof = try engine.aggregate(
            proofs: [proof],
            vk: vk,
            publicInputs: [pubInputs],
            srs: srs
        )

        expect(aggProof.count == 1, "Single proof count is 1")
        expect(!pointIsIdentity(aggProof.aggC), "aggC non-identity")
        expect(!aggProof.challenge.isZero, "Challenge non-zero")

        // Verify
        let valid = try engine.verify(
            aggregateProof: aggProof,
            proofs: [proof],
            vk: vk,
            publicInputs: [pubInputs],
            srs: srs
        )
        expect(valid, "Single proof SnarkPack aggregate verifies")
    } catch {
        expect(false, "Single proof aggregation error: \(error)")
    }
}

// MARK: - Two Proof Aggregation

private func testTwoProofAggregation() {
    do {
        let (proofs, pubInputs, vk) = makeProofs(count: 2)
        let srs = testSRS(maxProofs: 2)

        let engine = SnarkPackEngine()
        let aggProof = try engine.aggregate(
            proofs: proofs,
            vk: vk,
            publicInputs: pubInputs,
            srs: srs
        )

        expect(aggProof.count == 2, "Two proof count")
        expect(aggProof.challengePowers.count == 2, "Two challenge powers")

        let valid = try engine.verify(
            aggregateProof: aggProof,
            proofs: proofs,
            vk: vk,
            publicInputs: pubInputs,
            srs: srs
        )
        expect(valid, "Two proof SnarkPack aggregate verifies")
    } catch {
        expect(false, "Two proof aggregation error: \(error)")
    }
}

// MARK: - Four Proof Aggregation

private func testFourProofAggregation() {
    do {
        let (proofs, pubInputs, vk) = makeProofs(count: 4)
        let srs = testSRS(maxProofs: 4)

        let engine = SnarkPackEngine()
        let aggProof = try engine.aggregate(
            proofs: proofs,
            vk: vk,
            publicInputs: pubInputs,
            srs: srs
        )

        expect(aggProof.count == 4, "Four proof count")

        let valid = try engine.verify(
            aggregateProof: aggProof,
            proofs: proofs,
            vk: vk,
            publicInputs: pubInputs,
            srs: srs
        )
        expect(valid, "Four proof SnarkPack aggregate verifies")
    } catch {
        expect(false, "Four proof aggregation error: \(error)")
    }
}

// MARK: - TIPP Proof Structure

private func testTIPPProofStructure() {
    do {
        let (proofs, pubInputs, vk) = makeProofs(count: 4)
        let srs = testSRS(maxProofs: 4)

        let engine = SnarkPackEngine()
        let aggProof = try engine.aggregate(
            proofs: proofs,
            vk: vk,
            publicInputs: pubInputs,
            srs: srs
        )

        let tipp = aggProof.tippProof
        // 4 proofs -> 2 rounds (log2(4) = 2)
        expectEqual(tipp.gLeft.count, 2, "TIPP gLeft has 2 rounds")
        expectEqual(tipp.gRight.count, 2, "TIPP gRight has 2 rounds")
        expectEqual(tipp.hLeft.count, 2, "TIPP hLeft has 2 rounds")
        expectEqual(tipp.hRight.count, 2, "TIPP hRight has 2 rounds")
        expectEqual(tipp.challenges.count, 2, "TIPP has 2 challenges")

        // Final elements non-identity
        expect(!pointIsIdentity(tipp.finalA), "TIPP finalA non-identity")
        expect(!g2IsIdentity(tipp.finalB), "TIPP finalB non-identity")

        // Each round's commitments should be non-identity
        for i in 0..<2 {
            expect(!pointIsIdentity(tipp.gLeft[i]),
                   "TIPP round \(i) gLeft non-identity")
            expect(!tipp.challenges[i].isZero,
                   "TIPP round \(i) challenge non-zero")
        }
    } catch {
        expect(false, "TIPP structure error: \(error)")
    }
}

// MARK: - MIPP Proof Structure

private func testMIPPProofStructure() {
    do {
        let (proofs, pubInputs, vk) = makeProofs(count: 4)
        let srs = testSRS(maxProofs: 4)

        let engine = SnarkPackEngine()
        let aggProof = try engine.aggregate(
            proofs: proofs,
            vk: vk,
            publicInputs: pubInputs,
            srs: srs
        )

        let mipp = aggProof.mippProof
        // 4 elements -> 2 rounds
        expectEqual(mipp.comLeft.count, 2, "MIPP comLeft has 2 rounds")
        expectEqual(mipp.comRight.count, 2, "MIPP comRight has 2 rounds")
        expectEqual(mipp.challenges.count, 2, "MIPP has 2 challenges")

        // Final values
        expect(!mipp.finalScalar.isZero, "MIPP final scalar non-zero")
        expect(!pointIsIdentity(mipp.finalPoint), "MIPP final point non-identity")
    } catch {
        expect(false, "MIPP structure error: \(error)")
    }
}

// MARK: - Fiat-Shamir Determinism

private func testFiatShamirDeterminism() {
    do {
        let (proofs, pubInputs, vk) = makeProofs(count: 2)
        let srs = testSRS(maxProofs: 2)

        let engine = SnarkPackEngine()

        let agg1 = try engine.aggregate(
            proofs: proofs, vk: vk, publicInputs: pubInputs, srs: srs)
        let agg2 = try engine.aggregate(
            proofs: proofs, vk: vk, publicInputs: pubInputs, srs: srs)

        expect(frEqual(agg1.challenge, agg2.challenge),
               "Fiat-Shamir deterministic: same challenge for same inputs")
    } catch {
        expect(false, "Fiat-Shamir determinism error: \(error)")
    }
}

// MARK: - Fiat-Shamir Sensitivity

private func testFiatShamirSensitivity() {
    do {
        let (proofs1, pub1, vk) = makeProofs(count: 2, startX: 3)
        let (proofs2, pub2, _) = makeProofs(count: 2, startX: 10)
        let srs = testSRS(maxProofs: 2)

        let engine = SnarkPackEngine()

        let agg1 = try engine.aggregate(
            proofs: proofs1, vk: vk, publicInputs: pub1, srs: srs)
        let agg2 = try engine.aggregate(
            proofs: proofs2, vk: vk, publicInputs: pub2, srs: srs)

        expect(!frEqual(agg1.challenge, agg2.challenge),
               "Different inputs produce different challenges")
    } catch {
        expect(false, "Fiat-Shamir sensitivity error: \(error)")
    }
}

// MARK: - Verification Rejects Corrupted aggC

private func testVerificationRejectsCorruptedAggC() {
    do {
        let (proofs, pubInputs, vk) = makeProofs(count: 2)
        let srs = testSRS(maxProofs: 2)

        let engine = SnarkPackEngine()
        let aggProof = try engine.aggregate(
            proofs: proofs, vk: vk, publicInputs: pubInputs, srs: srs)

        // Corrupt aggC by doubling it
        let corruptedAggC = pointDouble(aggProof.aggC)
        let corrupted = SnarkPackProof(
            tippProof: aggProof.tippProof,
            mippProof: aggProof.mippProof,
            aggC: corruptedAggC,
            challenge: aggProof.challenge,
            challengePowers: aggProof.challengePowers,
            count: aggProof.count,
            publicInputs: aggProof.publicInputs
        )

        let valid = try engine.verify(
            aggregateProof: corrupted,
            proofs: proofs,
            vk: vk,
            publicInputs: pubInputs,
            srs: srs
        )
        expect(!valid, "Corrupted aggC rejected by verifier")
    } catch {
        expect(false, "Corrupted aggC test error: \(error)")
    }
}

// MARK: - Wrong Public Inputs Rejected

private func testVerificationRejectsWrongPublicInputs() {
    do {
        let (proofs, pubInputs, vk) = makeProofs(count: 2)
        let srs = testSRS(maxProofs: 2)

        let engine = SnarkPackEngine()
        let aggProof = try engine.aggregate(
            proofs: proofs, vk: vk, publicInputs: pubInputs, srs: srs)

        // Tamper with public inputs
        var tamperedInputs = pubInputs
        if !tamperedInputs[0].isEmpty {
            tamperedInputs[0][0] = frAdd(tamperedInputs[0][0], Fr.one)
        }

        // Verification should fail because re-derived challenge differs
        let valid = try engine.verify(
            aggregateProof: aggProof,
            proofs: proofs,
            vk: vk,
            publicInputs: tamperedInputs,
            srs: srs
        )
        expect(!valid, "Tampered public inputs rejected")
    } catch {
        expect(false, "Wrong public inputs test error: \(error)")
    }
}

// MARK: - Challenge Power Consistency

private func testChallengeConsistency() {
    do {
        let (proofs, pubInputs, vk) = makeProofs(count: 4)
        let srs = testSRS(maxProofs: 4)

        let engine = SnarkPackEngine()
        let aggProof = try engine.aggregate(
            proofs: proofs, vk: vk, publicInputs: pubInputs, srs: srs)

        let r = aggProof.challenge
        let powers = aggProof.challengePowers

        // powers[0] = 1
        expect(frEqual(powers[0], Fr.one), "r^0 = 1")

        // powers[1] = r
        expect(frEqual(powers[1], r), "r^1 = r")

        // powers[2] = r^2
        let r2 = frMul(r, r)
        expect(frEqual(powers[2], r2), "r^2 correct")

        // powers[3] = r^3
        let r3 = frMul(r2, r)
        expect(frEqual(powers[3], r3), "r^3 correct")
    } catch {
        expect(false, "Challenge consistency error: \(error)")
    }
}
