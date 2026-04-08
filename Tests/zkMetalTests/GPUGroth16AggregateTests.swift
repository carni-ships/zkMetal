// GPUGroth16AggregateEngine tests: GPU-accelerated Groth16 proof aggregation
import zkMetal

public func runGPUGroth16AggregateTests() {
    suite("GPU Groth16 Aggregate Engine")

    testSRSGeneration()
    testSingleProofAggregate()
    testTwoProofHomogeneousAggregate()
    testFourProofHomogeneousAggregate()
    testAggregateProofStructure()
    testFiatShamirDeterminism()
    testFiatShamirSensitivity()
    testIPPAStructure()
    testChallengePowerComputation()
    testChallengeSumConsistency()
    testVerificationRejectsCorruptedAggA()
    testVerificationRejectsCorruptedAggC()
    testVerificationRejectsWrongPublicInputs()
    testHeterogeneousTwoCircuits()
    testAggregateStatistics()
    testCPUFallbackPath()
    testAggregateAndBatchVerify()
    testSingleProofRoundTrip()
    testProofInputDescriptor()
    testSRSMaxProofsEnforced()
}

// MARK: - Helpers

/// Generate a Groth16 proof for the example circuit (x^3 + x + 5 = y) with given x.
private func generateTestProof(x: UInt64) -> (proof: Groth16Proof,
                                                vk: Groth16VerificationKey,
                                                pk: Groth16ProvingKey,
                                                publicInputs: [Fr],
                                                r1cs: R1CSInstance) {
    let r1cs = buildExampleCircuit()
    let (pubInputs, witness) = computeExampleWitness(x: x)
    let setup = Groth16Setup()
    let (pk, vk) = setup.setup(r1cs: r1cs)
    let prover = try! Groth16Prover()
    let proof = try! prover.prove(pk: pk, r1cs: r1cs, publicInputs: pubInputs, witness: witness)
    return (proof, vk, pk, pubInputs, r1cs)
}

/// Generate multiple proofs for the same circuit with different inputs.
private func generateMultipleProofs(count: Int, startX: UInt64 = 3) -> (
    proofs: [Groth16Proof],
    publicInputs: [[Fr]],
    vk: Groth16VerificationKey,
    pk: Groth16ProvingKey,
    r1cs: R1CSInstance
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

    return (proofs, allPubInputs, vk, pk, r1cs)
}

/// Create an AggregationSRS for testing.
private func testSRS(maxProofs: Int = 16) -> AggregationSRS {
    let seed = frFromInt(42)
    return AggregationSRS.generate(maxProofs: maxProofs, seed: seed)
}

// MARK: - SRS Tests

private func testSRSGeneration() {
    let srs = testSRS(maxProofs: 8)

    expect(srs.maxProofs == 8, "SRS maxProofs is 8")
    expect(srs.g1Powers.count == 8, "SRS has 8 G1 powers")

    // First power should be the generator (tau^0 * G1 = G1)
    let g1 = pointFromAffine(bn254G1Generator())
    let firstPower = srs.g1Powers[0]
    expect(pointEqual(firstPower, g1), "SRS g1Powers[0] is generator")

    // Powers should be distinct
    expect(!pointEqual(srs.g1Powers[0], srs.g1Powers[1]),
           "SRS g1Powers[0] != g1Powers[1]")
    expect(!pointEqual(srs.g1Powers[1], srs.g1Powers[2]),
           "SRS g1Powers[1] != g1Powers[2]")

    // G2 elements should be non-identity
    expect(!g2IsIdentity(srs.g2Gen), "SRS g2Gen non-identity")
    expect(!g2IsIdentity(srs.g2Tau), "SRS g2Tau non-identity")
}

// MARK: - Single Proof Aggregation

private func testSingleProofAggregate() {
    do {
        let (proof, vk, _, pubInputs, _) = generateTestProof(x: 3)
        let srs = testSRS()

        let engine = try GPUGroth16AggregateEngine()
        let aggProof = try engine.aggregateHomogeneous(
            proofs: [proof],
            publicInputs: [pubInputs],
            vk: vk,
            srs: srs
        )

        expect(aggProof.count == 1, "Single proof aggregate count is 1")
        expect(!pointIsIdentity(aggProof.aggA), "aggA non-identity")
        expect(!pointIsIdentity(aggProof.aggC), "aggC non-identity")

        // Verify
        let valid = engine.verifyHomogeneous(
            aggProof: aggProof,
            originalProofs: [proof],
            vk: vk,
            srs: srs
        )
        expect(valid, "Single proof aggregate verifies")
    } catch {
        expect(false, "Single proof aggregate error: \(error)")
    }
}

// MARK: - Two Proof Homogeneous

private func testTwoProofHomogeneousAggregate() {
    do {
        let (proofs, pubInputs, vk, _, _) = generateMultipleProofs(count: 2)
        let srs = testSRS()

        let engine = try GPUGroth16AggregateEngine()
        let aggProof = try engine.aggregateHomogeneous(
            proofs: proofs,
            publicInputs: pubInputs,
            vk: vk,
            srs: srs
        )

        expect(aggProof.count == 2, "Two proof aggregate count is 2")
        expect(aggProof.challengePowers.count == 2, "Two challenge powers")

        let valid = engine.verifyHomogeneous(
            aggProof: aggProof,
            originalProofs: proofs,
            vk: vk,
            srs: srs
        )
        expect(valid, "Two proof homogeneous aggregate verifies")
    } catch {
        expect(false, "Two proof aggregate error: \(error)")
    }
}

// MARK: - Four Proof Homogeneous

private func testFourProofHomogeneousAggregate() {
    do {
        let (proofs, pubInputs, vk, _, _) = generateMultipleProofs(count: 4)
        let srs = testSRS()

        let engine = try GPUGroth16AggregateEngine()
        let aggProof = try engine.aggregateHomogeneous(
            proofs: proofs,
            publicInputs: pubInputs,
            vk: vk,
            srs: srs
        )

        expect(aggProof.count == 4, "Four proof aggregate count")
        expect(aggProof.ippProof.leftCommitments.count == 2,
               "IPPA has 2 rounds for 4 proofs (log2(4))")

        let valid = engine.verifyHomogeneous(
            aggProof: aggProof,
            originalProofs: proofs,
            vk: vk,
            srs: srs
        )
        expect(valid, "Four proof homogeneous aggregate verifies")
    } catch {
        expect(false, "Four proof aggregate error: \(error)")
    }
}

// MARK: - Aggregate Proof Structure

private func testAggregateProofStructure() {
    do {
        let (proofs, pubInputs, vk, _, _) = generateMultipleProofs(count: 2)
        let srs = testSRS()

        let engine = try GPUGroth16AggregateEngine()
        let aggProof = try engine.aggregateHomogeneous(
            proofs: proofs,
            publicInputs: pubInputs,
            vk: vk,
            srs: srs
        )

        // Check all fields are populated
        expect(!aggProof.challenge.isZero, "Challenge is non-zero")
        expect(aggProof.challengePowers.count == 2, "Correct power count")
        expect(!aggProof.challengeSum.isZero, "Challenge sum non-zero")
        expect(aggProof.publicInputs.count == 2, "Public inputs count")
        expect(aggProof.vkIndices.count == 2, "VK indices count")
        expectEqual(aggProof.vkIndices[0], 0, "VK index 0 is 0")
        expectEqual(aggProof.vkIndices[1], 0, "VK index 1 is 0")

        // First challenge power should be 1 (r^0 = 1)
        expect(frEqual(aggProof.challengePowers[0], Fr.one),
               "First challenge power is 1")
    } catch {
        expect(false, "Aggregate structure error: \(error)")
    }
}

// MARK: - Fiat-Shamir Determinism

private func testFiatShamirDeterminism() {
    do {
        let (proofs, pubInputs, vk, _, _) = generateMultipleProofs(count: 2)
        let srs = testSRS()

        let engine = try GPUGroth16AggregateEngine()

        // Aggregate twice with same inputs
        let agg1 = try engine.aggregateHomogeneous(
            proofs: proofs, publicInputs: pubInputs, vk: vk, srs: srs)
        let agg2 = try engine.aggregateHomogeneous(
            proofs: proofs, publicInputs: pubInputs, vk: vk, srs: srs)

        // Challenges should be identical (deterministic Fiat-Shamir)
        expect(frEqual(agg1.challenge, agg2.challenge),
               "Fiat-Shamir produces same challenge for same inputs")
        expect(frEqual(agg1.challengeSum, agg2.challengeSum),
               "Challenge sums match for same inputs")
    } catch {
        expect(false, "Fiat-Shamir determinism error: \(error)")
    }
}

// MARK: - Fiat-Shamir Sensitivity

private func testFiatShamirSensitivity() {
    do {
        let (proofs2, pubInputs2, vk, _, _) = generateMultipleProofs(count: 2, startX: 3)
        let (proofs2b, pubInputs2b, _, _, _) = generateMultipleProofs(count: 2, startX: 10)
        let srs = testSRS()

        let engine = try GPUGroth16AggregateEngine()

        let agg1 = try engine.aggregateHomogeneous(
            proofs: proofs2, publicInputs: pubInputs2, vk: vk, srs: srs)
        let agg2 = try engine.aggregateHomogeneous(
            proofs: proofs2b, publicInputs: pubInputs2b, vk: vk, srs: srs)

        // Different inputs should produce different challenges
        expect(!frEqual(agg1.challenge, agg2.challenge),
               "Different inputs produce different Fiat-Shamir challenges")
    } catch {
        expect(false, "Fiat-Shamir sensitivity error: \(error)")
    }
}

// MARK: - IPPA Structure

private func testIPPAStructure() {
    do {
        let (proofs, pubInputs, vk, _, _) = generateMultipleProofs(count: 4)
        let srs = testSRS()

        let engine = try GPUGroth16AggregateEngine()
        let aggProof = try engine.aggregateHomogeneous(
            proofs: proofs, publicInputs: pubInputs, vk: vk, srs: srs)

        let ipp = aggProof.ippProof
        // 4 proofs -> 2 rounds of recursive halving
        expectEqual(ipp.leftCommitments.count, 2, "IPPA left commitments count")
        expectEqual(ipp.rightCommitments.count, 2, "IPPA right commitments count")
        expectEqual(ipp.challenges.count, 2, "IPPA challenges count")

        // Final point should be non-identity (unless very unlikely cancellation)
        expect(!pointIsIdentity(ipp.finalPoint), "IPPA final point non-identity")
        expect(!ipp.finalScalar.isZero, "IPPA final scalar non-zero")

        // Each round's L and R commitments should be non-identity
        for i in 0..<2 {
            expect(!pointIsIdentity(ipp.leftCommitments[i]),
                   "IPPA round \(i) left non-identity")
            expect(!pointIsIdentity(ipp.rightCommitments[i]),
                   "IPPA round \(i) right non-identity")
            expect(!ipp.challenges[i].isZero,
                   "IPPA round \(i) challenge non-zero")
        }
    } catch {
        expect(false, "IPPA structure error: \(error)")
    }
}

// MARK: - Challenge Power Computation

private func testChallengePowerComputation() {
    do {
        let (proofs, pubInputs, vk, _, _) = generateMultipleProofs(count: 4)
        let srs = testSRS()

        let engine = try GPUGroth16AggregateEngine()
        let aggProof = try engine.aggregateHomogeneous(
            proofs: proofs, publicInputs: pubInputs, vk: vk, srs: srs)

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
        expect(false, "Challenge power error: \(error)")
    }
}

// MARK: - Challenge Sum Consistency

private func testChallengeSumConsistency() {
    do {
        let (proofs, pubInputs, vk, _, _) = generateMultipleProofs(count: 4)
        let srs = testSRS()

        let engine = try GPUGroth16AggregateEngine()
        let aggProof = try engine.aggregateHomogeneous(
            proofs: proofs, publicInputs: pubInputs, vk: vk, srs: srs)

        // Manually compute sum of powers
        var manualSum = Fr.zero
        for p in aggProof.challengePowers {
            manualSum = frAdd(manualSum, p)
        }

        expect(frEqual(manualSum, aggProof.challengeSum),
               "Challenge sum matches manual computation")
    } catch {
        expect(false, "Challenge sum error: \(error)")
    }
}

// MARK: - Verification Rejects Corrupted aggA

private func testVerificationRejectsCorruptedAggA() {
    do {
        let (proofs, pubInputs, vk, _, _) = generateMultipleProofs(count: 2)
        let srs = testSRS()

        let engine = try GPUGroth16AggregateEngine()
        let aggProof = try engine.aggregateHomogeneous(
            proofs: proofs, publicInputs: pubInputs, vk: vk, srs: srs)

        // Corrupt aggA by doubling it
        let corruptedAggA = pointDouble(aggProof.aggA)
        let corrupted = GPUAggregatedProof(
            aggA: corruptedAggA,
            aggC: aggProof.aggC,
            aggL: aggProof.aggL,
            ippProof: aggProof.ippProof,
            challenge: aggProof.challenge,
            challengePowers: aggProof.challengePowers,
            challengeSum: aggProof.challengeSum,
            count: aggProof.count,
            publicInputs: aggProof.publicInputs,
            vkIndices: aggProof.vkIndices
        )

        let valid = engine.verifyHomogeneous(
            aggProof: corrupted, originalProofs: proofs, vk: vk, srs: srs)
        expect(!valid, "Corrupted aggA rejected by verifier")
    } catch {
        expect(false, "Corrupted aggA test error: \(error)")
    }
}

// MARK: - Verification Rejects Corrupted aggC

private func testVerificationRejectsCorruptedAggC() {
    do {
        let (proofs, pubInputs, vk, _, _) = generateMultipleProofs(count: 2)
        let srs = testSRS()

        let engine = try GPUGroth16AggregateEngine()
        let aggProof = try engine.aggregateHomogeneous(
            proofs: proofs, publicInputs: pubInputs, vk: vk, srs: srs)

        // Corrupt aggC
        let corruptedAggC = pointDouble(aggProof.aggC)
        let corrupted = GPUAggregatedProof(
            aggA: aggProof.aggA,
            aggC: corruptedAggC,
            aggL: aggProof.aggL,
            ippProof: aggProof.ippProof,
            challenge: aggProof.challenge,
            challengePowers: aggProof.challengePowers,
            challengeSum: aggProof.challengeSum,
            count: aggProof.count,
            publicInputs: aggProof.publicInputs,
            vkIndices: aggProof.vkIndices
        )

        let valid = engine.verifyHomogeneous(
            aggProof: corrupted, originalProofs: proofs, vk: vk, srs: srs)
        expect(!valid, "Corrupted aggC rejected by verifier")
    } catch {
        expect(false, "Corrupted aggC test error: \(error)")
    }
}

// MARK: - Wrong Public Inputs Rejected

private func testVerificationRejectsWrongPublicInputs() {
    do {
        let (proofs, pubInputs, vk, _, _) = generateMultipleProofs(count: 2)
        let srs = testSRS()

        let engine = try GPUGroth16AggregateEngine()
        let aggProof = try engine.aggregateHomogeneous(
            proofs: proofs, publicInputs: pubInputs, vk: vk, srs: srs)

        // Tamper with public inputs in the aggregated proof
        var tamperedInputs = aggProof.publicInputs
        if !tamperedInputs[0].isEmpty {
            tamperedInputs[0][0] = frAdd(tamperedInputs[0][0], Fr.one)
        }
        let tampered = GPUAggregatedProof(
            aggA: aggProof.aggA,
            aggC: aggProof.aggC,
            aggL: aggProof.aggL,
            ippProof: aggProof.ippProof,
            challenge: aggProof.challenge,
            challengePowers: aggProof.challengePowers,
            challengeSum: aggProof.challengeSum,
            count: aggProof.count,
            publicInputs: tamperedInputs,
            vkIndices: aggProof.vkIndices
        )

        // Re-derived challenge will differ since public inputs are in transcript
        let valid = engine.verifyHomogeneous(
            aggProof: tampered, originalProofs: proofs, vk: vk, srs: srs)
        expect(!valid, "Tampered public inputs rejected")
    } catch {
        expect(false, "Wrong public inputs test error: \(error)")
    }
}

// MARK: - Heterogeneous Two Circuits

private func testHeterogeneousTwoCircuits() {
    do {
        // Generate proofs from two different circuit setups
        let (proof1, vk1, _, pub1, _) = generateTestProof(x: 3)
        let (proof2, vk2, _, pub2, _) = generateTestProof(x: 7)

        let srs = testSRS()
        let engine = try GPUGroth16AggregateEngine()

        let inputs = [
            AggregateProofInput(proof: proof1, publicInputs: pub1, vk: vk1, circuitIndex: 0),
            AggregateProofInput(proof: proof2, publicInputs: pub2, vk: vk2, circuitIndex: 1),
        ]

        let aggProof = try engine.aggregate(inputs: inputs, srs: srs)

        expect(aggProof.count == 2, "Heterogeneous aggregate count")
        expectEqual(aggProof.vkIndices[0], 0, "First proof circuit index 0")
        expectEqual(aggProof.vkIndices[1], 1, "Second proof circuit index 1")

        let vks: [Int: Groth16VerificationKey] = [0: vk1, 1: vk2]
        let valid = engine.verify(
            aggProof: aggProof,
            originalProofs: [proof1, proof2],
            vks: vks,
            srs: srs
        )
        expect(valid, "Heterogeneous two-circuit aggregate verifies")
    } catch {
        expect(false, "Heterogeneous test error: \(error)")
    }
}

// MARK: - Aggregate Statistics

private func testAggregateStatistics() {
    let r1cs = buildExampleCircuit()
    let setup = Groth16Setup()
    let (_, vk) = setup.setup(r1cs: r1cs)
    let (pubInputs, _) = computeExampleWitness(x: 3)
    let dummyProof = Groth16Proof(
        a: pointFromAffine(bn254G1Generator()),
        b: g2FromAffine(bn254G2Generator()),
        c: pointFromAffine(bn254G1Generator())
    )

    let inputs = (0..<8).map { i in
        AggregateProofInput(
            proof: dummyProof,
            publicInputs: pubInputs,
            vk: vk,
            circuitIndex: i < 4 ? 0 : 1
        )
    }

    let stats = AggregateStatistics.compute(inputs: inputs, gpuThreshold: 64)
    expectEqual(stats.proofCount, 8, "Stats proof count")
    expectEqual(stats.circuitCount, 2, "Stats circuit count")
    expectEqual(stats.ippaRounds, 3, "Stats IPPA rounds (log2(8) = 3)")
    expect(!stats.usedGPU, "8 proofs below GPU threshold of 64")

    let statsGPU = AggregateStatistics.compute(inputs: inputs, gpuThreshold: 4)
    expect(statsGPU.usedGPU, "8 proofs above GPU threshold of 4")
}

// MARK: - CPU Fallback Path

private func testCPUFallbackPath() {
    do {
        let (proofs, pubInputs, vk, _, _) = generateMultipleProofs(count: 2)
        let srs = testSRS()

        let engine = try GPUGroth16AggregateEngine()
        // Set threshold very high to force CPU path
        engine.gpuMSMThreshold = 1000

        let aggProof = try engine.aggregateHomogeneous(
            proofs: proofs, publicInputs: pubInputs, vk: vk, srs: srs)

        let valid = engine.verifyHomogeneous(
            aggProof: aggProof, originalProofs: proofs, vk: vk, srs: srs)
        expect(valid, "CPU fallback path produces valid aggregate")
    } catch {
        expect(false, "CPU fallback error: \(error)")
    }
}

// MARK: - Batch Verify Convenience

private func testAggregateAndBatchVerify() {
    do {
        // Use generateMultipleProofs so both proofs share the same setup (pk/vk).
        // generateTestProof creates independent setups, so proof2's pk wouldn't
        // match vk from proof1's setup.
        let (proofs, pubInputs, vk, _, _) = generateMultipleProofs(count: 2, startX: 5)
        let srs = testSRS()

        let engine = try GPUGroth16AggregateEngine()
        let inputs = [
            AggregateProofInput(proof: proofs[0], publicInputs: pubInputs[0], vk: vk, circuitIndex: 0),
            AggregateProofInput(proof: proofs[1], publicInputs: pubInputs[1], vk: vk, circuitIndex: 0),
        ]

        let valid = try engine.batchVerify(inputs: inputs, srs: srs)
        expect(valid, "Batch verify convenience method works")
    } catch {
        expect(false, "Batch verify error: \(error)")
    }
}

// MARK: - Single Proof Round Trip

private func testSingleProofRoundTrip() {
    do {
        let (proof, vk, _, pubInputs, _) = generateTestProof(x: 11)
        let srs = testSRS()

        // Verify original proof first
        let verifier = Groth16Verifier()
        let origValid = verifier.verify(proof: proof, vk: vk, publicInputs: pubInputs)
        expect(origValid, "Original proof is valid before aggregation")

        // Now aggregate and verify
        let engine = try GPUGroth16AggregateEngine()
        let input = AggregateProofInput(
            proof: proof, publicInputs: pubInputs, vk: vk, circuitIndex: 0)
        let valid = try engine.batchVerify(inputs: [input], srs: srs)
        expect(valid, "Single proof round trip through aggregation verifies")
    } catch {
        expect(false, "Round trip error: \(error)")
    }
}

// MARK: - Proof Input Descriptor

private func testProofInputDescriptor() {
    let r1cs = buildExampleCircuit()
    let setup = Groth16Setup()
    let (_, vk) = setup.setup(r1cs: r1cs)
    let (pubInputs, _) = computeExampleWitness(x: 3)
    let dummyProof = Groth16Proof(
        a: pointFromAffine(bn254G1Generator()),
        b: g2FromAffine(bn254G2Generator()),
        c: pointFromAffine(bn254G1Generator())
    )

    let input = AggregateProofInput(
        proof: dummyProof,
        publicInputs: pubInputs,
        vk: vk,
        circuitIndex: 42
    )

    expectEqual(input.circuitIndex, 42, "Circuit index preserved")
    expect(input.publicInputs.count == pubInputs.count, "Public inputs preserved")
    expect(!pointIsIdentity(input.proof.a), "Proof A non-identity")
}

// MARK: - SRS Max Proofs Enforced

private func testSRSMaxProofsEnforced() {
    let srs = testSRS(maxProofs: 4)
    expectEqual(srs.maxProofs, 4, "SRS maxProofs is 4")
    expectEqual(srs.g1Powers.count, 4, "SRS has exactly 4 G1 powers")

    // Verify powers are on the curve (non-identity, distinct)
    for i in 0..<4 {
        expect(!pointIsIdentity(srs.g1Powers[i]),
               "SRS power \(i) non-identity")
    }
    for i in 0..<3 {
        expect(!pointEqual(srs.g1Powers[i], srs.g1Powers[i + 1]),
               "SRS power \(i) != power \(i+1)")
    }
}
