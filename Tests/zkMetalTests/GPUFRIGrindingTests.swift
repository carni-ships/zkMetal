// GPU FRI Grinding Engine tests
// Validates proof-of-work grinding: Poseidon2/Keccak hash, difficulty targets,
// leading zero counting, batch verification, FRI integration, multi-round grinding,
// nonce search strategies, adaptive difficulty, and transcript helpers.
// Run: swift build && .build/debug/zkMetalTests

import zkMetal
import Foundation

// MARK: - Test RNG

private struct GrindRNG {
    var state: UInt64

    mutating func next32() -> UInt32 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return UInt32(state >> 33)
    }

    mutating func nextFr() -> Fr {
        let raw = Fr(v: (next32() & 0x0FFFFFFF, next32(), next32(), next32(),
                         next32(), next32(), next32(), next32() & 0x0FFFFFFF))
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }

    mutating func nextNonZeroFr() -> Fr {
        var v = nextFr()
        while v.isZero { v = nextFr() }
        return v
    }

    mutating func nextUInt64() -> UInt64 {
        return UInt64(next32()) | (UInt64(next32()) << 32)
    }
}

// MARK: - Leading Zeros Helper Tests

private func testLeadingZeroCount() {
    suite("FRI Grinding: Leading zero count")

    // Zero should have 256 leading zeros (all limbs are zero)
    let lzZero = GrindingVerifier.countLeadingZeros(Fr.zero)
    expectEqual(lzZero, 256, "Fr.zero should have 256 leading zeros")

    // Fr.one (Montgomery form) should have far fewer leading zeros
    let lzOne = GrindingVerifier.countLeadingZeros(Fr.one)
    expect(lzOne < 256, "Fr.one should have fewer than 256 leading zeros")
    expect(lzOne >= 0, "Leading zeros should be non-negative")

    // Construct a value with exactly 64 leading zeros (limbs[3] = 0, limbs[2] has MSB set)
    // Note: from64 treats each element as a full UInt64, so the upper 32 bits must be non-zero
    // for the 64-bit limb to have 0 leading zeros.
    let val64 = Fr.from64([0xDEADBEEF, 0x12345678, 0xABCDEF0100000000, 0])
    let lz64 = GrindingVerifier.countLeadingZeros(val64)
    expectEqual(lz64, 64, "Value with zero MSB limb should have 64 leading zeros")

    // Value with limbs[3] = 1 should have 63 leading zeros
    let val63 = Fr.from64([0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 1])
    let lz63 = GrindingVerifier.countLeadingZeros(val63)
    expectEqual(lz63, 63, "Value with MSB limb = 1 should have 63 leading zeros")

    // Value with limbs[3] = 0x8000000000000000 should have 0 leading zeros
    let valMax = Fr.from64([0, 0, 0, 0x8000000000000000])
    let lzMax = GrindingVerifier.countLeadingZeros(valMax)
    expectEqual(lzMax, 0, "Value with MSB set should have 0 leading zeros")

    // Value with limbs[3] = 0, limbs[2] = 0, limbs[1] = 0x100 should have 128 + 55 = 183
    let val183 = Fr.from64([0xFFFF, 0x100, 0, 0])
    let lz183 = GrindingVerifier.countLeadingZeros(val183)
    // limbs[3]=0: +64, limbs[2]=0: +64, limbs[1]=0x100: leadingZeroBitCount of 0x100 = 55
    expectEqual(lz183, 64 + 64 + 55,
                "Three zero limbs + 0x100 should have 183 leading zeros")

    // All zero limbs except first (LSB)
    let valLSB = Fr.from64([1, 0, 0, 0])
    let lzLSB = GrindingVerifier.countLeadingZeros(valLSB)
    expectEqual(lzLSB, 64 * 3 + 63, "Only LSB set should have 255 leading zeros")
}

// MARK: - Difficulty Target Tests

private func testMeetsTarget() {
    suite("FRI Grinding: Meets target check")

    // Zero meets any difficulty
    expect(GrindingVerifier.meetsTarget(hash: Fr.zero, difficulty: 1),
           "Zero should meet difficulty 1")
    expect(GrindingVerifier.meetsTarget(hash: Fr.zero, difficulty: 64),
           "Zero should meet difficulty 64")

    // Construct a value with exactly N leading zeros and verify
    // 128 leading zeros: limbs[3]=0, limbs[2]=0, limbs[1] has bit set
    let val128 = Fr.from64([0xFFFF, 0x8000000000000000, 0, 0])
    let lz128 = GrindingVerifier.countLeadingZeros(val128)
    expectEqual(lz128, 128, "val128 should have exactly 128 leading zeros")
    expect(GrindingVerifier.meetsTarget(hash: val128, difficulty: 128),
           "val128 should meet difficulty 128")
    expect(!GrindingVerifier.meetsTarget(hash: val128, difficulty: 129),
           "val128 should NOT meet difficulty 129")

    // Value with 1 leading zero
    let val1 = Fr.from64([0, 0, 0, 0x4000000000000000])
    expect(GrindingVerifier.meetsTarget(hash: val1, difficulty: 1),
           "val1 should meet difficulty 1")
    expect(!GrindingVerifier.meetsTarget(hash: val1, difficulty: 2),
           "val1 should NOT meet difficulty 2")

    // Random field elements typically have very few leading zeros
    var rng = GrindRNG(state: 99)
    var lowLeadingCount = 0
    for _ in 0..<20 {
        let v = rng.nextFr()
        let lz = GrindingVerifier.countLeadingZeros(v)
        if lz < 8 { lowLeadingCount += 1 }
    }
    // Most random Fr values should have < 8 leading zeros
    expect(lowLeadingCount > 10,
           "Most random Fr values should have fewer than 8 leading zeros")
}

// MARK: - Poseidon2 Grinding Tests

private func testPoseidon2Grinding() {
    suite("FRI Grinding: Poseidon2 basic grind")

    do {
        let engine = try GPUFRIGrindingEngine()

        // Low difficulty: should find nonce quickly
        let seed = frFromInt(42)
        let config = GrindingConfig(difficulty: 4, hashFunction: .poseidon2,
                                    maxNonces: 1 << 16, batchSize: 1 << 12,
                                    cpuVerify: true)

        let result = try engine.grind(seed: seed, config: config)

        // The nonce should produce a hash with >= 4 leading zeros
        let hash = GrindingVerifier.computeHash(seed: seed, nonce: result.nonce,
                                                 hashFunction: .poseidon2)
        let lz = GrindingVerifier.countLeadingZeros(hash)
        expect(lz >= 4, "Found nonce should have >= 4 leading zeros, got \(lz)")

        // Result should have been CPU verified
        expect(result.cpuVerified, "Result should be CPU verified")

        // Hash rate should be positive
        expect(result.hashRate > 0, "Hash rate should be positive")

        // Nonces checked should be >= 1
        expect(result.noncesChecked >= 1, "Should have checked at least 1 nonce")

    } catch {
        expect(false, "Poseidon2 grinding failed: \(error)")
    }
}

private func testPoseidon2GrindingDeterminism() {
    suite("FRI Grinding: Poseidon2 determinism")

    // Same seed should produce the same nonce (deterministic search)
    do {
        let engine = try GPUFRIGrindingEngine()
        let seed = frFromInt(12345)
        let config = GrindingConfig(difficulty: 3, hashFunction: .poseidon2,
                                    maxNonces: 1 << 16, batchSize: 1 << 12,
                                    cpuVerify: false)

        let result1 = try engine.grind(seed: seed, config: config)
        let result2 = try engine.grind(seed: seed, config: config)

        expectEqual(result1.nonce, result2.nonce,
                    "Same seed should produce same nonce")

    } catch {
        expect(false, "Determinism test failed: \(error)")
    }
}

private func testPoseidon2GrindingDifferentSeeds() {
    suite("FRI Grinding: Poseidon2 different seeds")

    do {
        let engine = try GPUFRIGrindingEngine()
        let config = GrindingConfig(difficulty: 4, hashFunction: .poseidon2,
                                    maxNonces: 1 << 18, batchSize: 1 << 14,
                                    cpuVerify: true)

        var nonces = Set<UInt64>()
        var rng = GrindRNG(state: 777)

        for _ in 0..<5 {
            let seed = rng.nextFr()
            let result = try engine.grind(seed: seed, config: config)
            nonces.insert(result.nonce)

            let hash = result.hashOutput
            let lz = GrindingVerifier.countLeadingZeros(hash)
            expect(lz >= 4, "Each result should meet difficulty 4, got \(lz)")
        }

        // Different seeds should generally produce different nonces
        // (not guaranteed, but highly probable)
        expect(nonces.count >= 2,
               "Different seeds should produce different nonces (got \(nonces.count) unique)")

    } catch {
        expect(false, "Different seeds test failed: \(error)")
    }
}

// MARK: - Keccak Grinding Tests

private func testKeccakGrinding() {
    suite("FRI Grinding: Keccak-256 basic grind")

    do {
        let engine = try GPUFRIGrindingEngine()
        let seed = frFromInt(0xCAFEBABE)
        let config = GrindingConfig(difficulty: 4, hashFunction: .keccak256,
                                    maxNonces: 1 << 18, batchSize: 1 << 12,
                                    cpuVerify: true)

        let result = try engine.grind(seed: seed, config: config)

        // Verify the hash meets the target
        let hash = GrindingVerifier.computeHash(seed: seed, nonce: result.nonce,
                                                 hashFunction: .keccak256)
        let lz = GrindingVerifier.countLeadingZeros(hash)
        expect(lz >= 4, "Keccak nonce should have >= 4 leading zeros, got \(lz)")

        expect(result.cpuVerified, "Keccak result should be CPU verified")

    } catch {
        expect(false, "Keccak grinding failed: \(error)")
    }
}

private func testKeccakVsPoseidon2() {
    suite("FRI Grinding: Keccak vs Poseidon2 produce different results")

    do {
        let engine = try GPUFRIGrindingEngine()
        let seed = frFromInt(999)

        let posConfig = GrindingConfig(difficulty: 3, hashFunction: .poseidon2,
                                       maxNonces: 1 << 16, batchSize: 1 << 12,
                                       cpuVerify: true)
        let kecConfig = GrindingConfig(difficulty: 3, hashFunction: .keccak256,
                                       maxNonces: 1 << 16, batchSize: 1 << 12,
                                       cpuVerify: true)

        let posResult = try engine.grind(seed: seed, config: posConfig)
        let kecResult = try engine.grind(seed: seed, config: kecConfig)

        // The hash outputs should be different (different hash functions)
        let posHash = posResult.hashOutput
        let kecHash = kecResult.hashOutput
        let posLimbs = posHash.to64()
        let kecLimbs = kecHash.to64()
        var different = false
        for i in 0..<4 {
            if posLimbs[i] != kecLimbs[i] { different = true; break }
        }
        expect(different, "Different hash functions should produce different outputs")

    } catch {
        expect(false, "Keccak vs Poseidon2 test failed: \(error)")
    }
}

// MARK: - Grinding Proof Tests

private func testGrindingProof() {
    suite("FRI Grinding: Proof creation and verification")

    do {
        let engine = try GPUFRIGrindingEngine()
        let seed = frFromInt(0xBEEF)

        let proof = try engine.grindProof(seed: seed,
                                          config: GrindingConfig(difficulty: 4,
                                                                  hashFunction: .poseidon2,
                                                                  maxNonces: 1 << 18,
                                                                  batchSize: 1 << 12))

        // Proof should be verifiable
        expect(proof.verify(), "Grinding proof should verify")

        // Proof fields should be correct
        let seedLimbs = proof.seed.to64()
        let expectedLimbs = seed.to64()
        var seedMatch = true
        for i in 0..<4 {
            if seedLimbs[i] != expectedLimbs[i] { seedMatch = false; break }
        }
        expect(seedMatch, "Proof seed should match input seed")
        expectEqual(proof.difficulty, 4, "Proof difficulty should be 4")

        // Compute hash and verify leading zeros
        let hash = proof.computeHash()
        let lz = GrindingVerifier.countLeadingZeros(hash)
        expect(lz >= 4, "Proof hash should have >= 4 leading zeros, got \(lz)")

    } catch {
        expect(false, "Grinding proof test failed: \(error)")
    }
}

private func testInvalidProof() {
    suite("FRI Grinding: Invalid proof detection")

    // Create a valid proof, then tamper with it
    let seed = frFromInt(42)
    let fakeSeed = frFromInt(43)

    // A proof with wrong seed should not verify for the original seed
    let fakeProof = GrindingProof(seed: fakeSeed, nonce: 0,
                                   difficulty: 16, hashFunction: .poseidon2)
    // This won't verify because nonce 0 is unlikely to meet difficulty 16
    let hash = fakeProof.computeHash()
    let lz = GrindingVerifier.countLeadingZeros(hash)
    if lz < 16 {
        expect(!fakeProof.verify(), "Fake proof should not verify at difficulty 16")
    }

    // A proof with zero nonce at very high difficulty should almost certainly fail
    let hardProof = GrindingProof(seed: seed, nonce: 0,
                                   difficulty: 60, hashFunction: .poseidon2)
    expect(!hardProof.verify(),
           "Nonce 0 should not meet difficulty 60")
}

// MARK: - Batch Verification Tests

private func testBatchVerification() {
    suite("FRI Grinding: Batch verification")

    do {
        let engine = try GPUFRIGrindingEngine()
        var rng = GrindRNG(state: 555)

        var proofs = [GrindingProof]()
        for _ in 0..<5 {
            let seed = rng.nextFr()
            let proof = try engine.grindProof(
                seed: seed,
                config: GrindingConfig(difficulty: 3, hashFunction: .poseidon2,
                                       maxNonces: 1 << 16, batchSize: 1 << 12))
            proofs.append(proof)
        }

        // All proofs should verify
        let results = GrindingVerifier.batchVerify(proofs: proofs)
        expectEqual(results.count, 5, "Should have 5 verification results")
        for (i, valid) in results.enumerated() {
            expect(valid, "Proof \(i) should verify")
        }

        // Detailed batch verification
        let (allValid, details) = GrindingVerifier.batchVerifyDetailed(proofs: proofs)
        expect(allValid, "All proofs should be valid")
        expectEqual(details.count, 5, "Should have 5 detailed results")
        for detail in details {
            expect(detail.valid, "Proof \(detail.index) should be valid in detailed check")
            expect(detail.leadingZeros >= 3,
                   "Proof \(detail.index) should have >= 3 leading zeros")
        }

    } catch {
        expect(false, "Batch verification test failed: \(error)")
    }
}

private func testBatchVerificationWithInvalid() {
    suite("FRI Grinding: Batch verification with invalid proof")

    do {
        let engine = try GPUFRIGrindingEngine()

        let validProof = try engine.grindProof(
            seed: frFromInt(100),
            config: GrindingConfig(difficulty: 3, hashFunction: .poseidon2,
                                   maxNonces: 1 << 16, batchSize: 1 << 12))

        // Create an obviously invalid proof
        let invalidProof = GrindingProof(seed: frFromInt(200), nonce: 12345,
                                          difficulty: 50, hashFunction: .poseidon2)

        let results = GrindingVerifier.batchVerify(proofs: [validProof, invalidProof])
        expectEqual(results.count, 2, "Should have 2 results")
        expect(results[0], "First proof should be valid")
        expect(!results[1], "Second proof should be invalid")

        let (allValid, _) = GrindingVerifier.batchVerifyDetailed(
            proofs: [validProof, invalidProof])
        expect(!allValid, "Not all proofs should be valid")

    } catch {
        expect(false, "Batch with invalid test failed: \(error)")
    }
}

// MARK: - FRI Integration Tests

private func testGrindingSeedComputation() {
    suite("FRI Grinding: Seed computation from FRI data")

    var rng = GrindRNG(state: 111)

    // Create mock commit roots and challenges
    var commitRoots = [Fr]()
    var challenges = [Fr]()
    for _ in 0..<5 {
        commitRoots.append(rng.nextFr())
        challenges.append(rng.nextFr())
    }

    // Seed should be deterministic
    let seed1 = FRIGrindingIntegration.computeGrindingSeed(
        commitRoots: commitRoots, foldingChallenges: challenges)
    let seed2 = FRIGrindingIntegration.computeGrindingSeed(
        commitRoots: commitRoots, foldingChallenges: challenges)
    expect(frEqual(seed1, seed2), "Same inputs should produce same seed")

    // Different inputs should produce different seeds
    var altRoots = commitRoots
    altRoots[0] = rng.nextFr()
    let seed3 = FRIGrindingIntegration.computeGrindingSeed(
        commitRoots: altRoots, foldingChallenges: challenges)
    expect(!frEqual(seed1, seed3), "Different roots should produce different seed")

    // Seed should not be zero
    expect(!seed1.isZero, "Grinding seed should not be zero")
}

private func testQueryDerivation() {
    suite("FRI Grinding: Query index derivation from proof")

    do {
        let engine = try GPUFRIGrindingEngine()
        let seed = frFromInt(0xDEAD)

        let proof = try engine.grindProof(
            seed: seed,
            config: GrindingConfig(difficulty: 3, hashFunction: .poseidon2,
                                   maxNonces: 1 << 16, batchSize: 1 << 12))

        let domainSize = 1024
        let numQueries = 16

        let indices = FRIGrindingIntegration.deriveQueryIndices(
            from: proof, domainSize: domainSize, numQueries: numQueries)

        expectEqual(indices.count, numQueries, "Should derive \(numQueries) indices")

        // All indices should be in [0, domainSize)
        for (i, idx) in indices.enumerated() {
            expect(idx >= 0 && idx < domainSize,
                   "Index \(i) should be in [0, \(domainSize)), got \(idx)")
        }

        // Indices should be deterministic
        let indices2 = FRIGrindingIntegration.deriveQueryIndices(
            from: proof, domainSize: domainSize, numQueries: numQueries)
        for i in 0..<numQueries {
            expectEqual(indices[i], indices2[i],
                        "Index \(i) should be deterministic")
        }

        // Check that indices have reasonable distribution (not all the same)
        let uniqueIndices = Set(indices)
        expect(uniqueIndices.count > numQueries / 2,
               "Should have reasonable spread of indices, got \(uniqueIndices.count) unique")

    } catch {
        expect(false, "Query derivation test failed: \(error)")
    }
}

// MARK: - Difficulty Estimator Tests

private func testDifficultyEstimator() {
    suite("FRI Grinding: Difficulty estimator")

    // Expected hashes for difficulty d is 2^d
    expectEqual(GrindingDifficultyEstimator.expectedHashes(difficulty: 0), 1,
                "2^0 = 1")
    expectEqual(GrindingDifficultyEstimator.expectedHashes(difficulty: 8), 256,
                "2^8 = 256")
    expectEqual(GrindingDifficultyEstimator.expectedHashes(difficulty: 16), 65536,
                "2^16 = 65536")
    expectEqual(GrindingDifficultyEstimator.expectedHashes(difficulty: 20), 1048576,
                "2^20 = 1048576")

    // Security bits = difficulty
    expectEqual(GrindingDifficultyEstimator.securityBits(difficulty: 16), 16,
                "Security bits should equal difficulty")

    // Success probability after 2^d tries should be ~63% (1 - 1/e)
    let prob = GrindingDifficultyEstimator.successProbability(
        difficulty: 8, tries: 256)
    expect(prob > 0.6 && prob < 0.7,
           "P(success in 2^d tries) should be ~63%, got \(prob)")

    // Success probability after 10*2^d tries should be very high
    let probHigh = GrindingDifficultyEstimator.successProbability(
        difficulty: 8, tries: 2560)
    expect(probHigh > 0.99,
           "P(success in 10x expected tries) should be > 99%, got \(probHigh)")

    // Tries for 99% probability
    let tries99 = GrindingDifficultyEstimator.triesForProbability(
        difficulty: 8, probability: 0.99)
    expect(tries99 > 256 && tries99 < 2000,
           "99% probability should need ~1177 tries for d=8, got \(tries99)")
}

private func testDifficultyRecommendation() {
    suite("FRI Grinding: Difficulty recommendation")

    // With 10M hashes/sec and 1 second budget, max difficulty ~23
    let rec1 = GrindingDifficultyEstimator.recommendDifficulty(
        securityBits: 16, maxTimeSeconds: 1.0, hashRate: 10_000_000)
    expect(rec1 != nil, "Should recommend a difficulty")
    if let d = rec1 {
        expectEqual(d, 16, "Should recommend exactly 16 bits for 16-bit security target")
    }

    // Infeasible: 32 bits security with only 1000 hashes/sec in 1 second
    let rec2 = GrindingDifficultyEstimator.recommendDifficulty(
        securityBits: 32, maxTimeSeconds: 1.0, hashRate: 1000)
    expect(rec2 == nil, "Should return nil for infeasible target")

    // Very fast device: 1B hashes/sec, 10 second budget
    let rec3 = GrindingDifficultyEstimator.recommendDifficulty(
        securityBits: 24, maxTimeSeconds: 10.0, hashRate: 1_000_000_000)
    expect(rec3 != nil, "Should recommend a difficulty for fast device")
    if let d = rec3 {
        expectEqual(d, 24, "Should recommend 24 bits when budget allows")
    }
}

// MARK: - Nonce Search Strategy Tests

private func testLinearNonceGenerator() {
    suite("FRI Grinding: Linear nonce generator")

    var gen = NonceGenerator(strategy: .linear)

    expectEqual(gen.next(), 0, "First nonce should be 0")
    expectEqual(gen.next(), 1, "Second nonce should be 1")
    expectEqual(gen.next(), 2, "Third nonce should be 2")

    // Batch generation
    let batch = gen.nextBatch(count: 5)
    expectEqual(batch.count, 5, "Batch should have 5 elements")
    expectEqual(batch[0], 3, "Batch should continue from 3")
    expectEqual(batch[4], 7, "Last batch element should be 7")
}

private func testStridedNonceGenerator() {
    suite("FRI Grinding: Strided nonce generator")

    var gen = NonceGenerator(strategy: .strided(offset: 10, stride: 3))
    expectEqual(gen.next(), 10, "First nonce should be offset=10")
    expectEqual(gen.next(), 13, "Second nonce should be 10+3=13")
    expectEqual(gen.next(), 16, "Third nonce should be 10+6=16")

    // 4 strided generators should partition [0..15]
    var gen0 = NonceGenerator(strategy: .strided(offset: 0, stride: 4))
    var gen1 = NonceGenerator(strategy: .strided(offset: 1, stride: 4))
    var gen2 = NonceGenerator(strategy: .strided(offset: 2, stride: 4))
    var gen3 = NonceGenerator(strategy: .strided(offset: 3, stride: 4))

    var allNonces = Set<UInt64>()
    for _ in 0..<4 {
        allNonces.insert(gen0.next()); allNonces.insert(gen1.next())
        allNonces.insert(gen2.next()); allNonces.insert(gen3.next())
    }
    expectEqual(allNonces.count, 16, "4 strided generators should cover all 16 nonces")
}

private func testRandomNonceGenerator() {
    suite("FRI Grinding: Random nonce generator")

    var gen = NonceGenerator(strategy: .random(seed: 42))

    var nonces = Set<UInt64>()
    for _ in 0..<100 {
        nonces.insert(gen.next())
    }

    // Random nonces should be mostly unique
    expect(nonces.count > 90, "Random nonces should be mostly unique, got \(nonces.count)")

    // Deterministic: same seed produces same sequence
    var gen2 = NonceGenerator(strategy: .random(seed: 42))
    var gen1replay = NonceGenerator(strategy: .random(seed: 42))

    for _ in 0..<10 {
        expectEqual(gen2.next(), gen1replay.next(),
                    "Same seed should produce same random sequence")
    }
}

// MARK: - Multi-Round Grinding Tests

private func testMultiRoundGrinding() {
    suite("FRI Grinding: Multi-round grinding")

    do {
        let engine = try GPUFRIGrindingEngine()
        var rng = GrindRNG(state: 333)

        // Simulate 3 FRI commit rounds
        var roundSeeds = [Fr]()
        for _ in 0..<3 {
            roundSeeds.append(rng.nextFr())
        }

        let config = MultiRoundGrinding.MultiRoundConfig(
            perRoundDifficulty: [3, 4, 3],
            hashFunction: .poseidon2,
            maxNoncesPerRound: 1 << 18)

        let result = try MultiRoundGrinding.grindAllRounds(
            roundSeeds: roundSeeds, config: config, engine: engine)

        expectEqual(result.nonces.count, 3, "Should have 3 nonces")
        expectEqual(result.hashOutputs.count, 3, "Should have 3 hash outputs")
        expect(result.totalNoncesChecked > 0, "Should have checked some nonces")
        expect(result.totalTimeSeconds >= 0, "Time should be non-negative")

        // Verify the multi-round result
        let verified = MultiRoundGrinding.verifyMultiRound(
            roundSeeds: roundSeeds, result: result, config: config)
        expect(verified, "Multi-round result should verify")

    } catch {
        expect(false, "Multi-round grinding failed: \(error)")
    }
}

private func testMultiRoundChaining() {
    suite("FRI Grinding: Multi-round nonce chaining")

    do {
        let engine = try GPUFRIGrindingEngine()
        var rng = GrindRNG(state: 444)

        var roundSeeds = [Fr]()
        for _ in 0..<2 {
            roundSeeds.append(rng.nextFr())
        }

        let config = MultiRoundGrinding.MultiRoundConfig(
            perRoundDifficulty: [3],  // Same difficulty repeated for all rounds
            hashFunction: .poseidon2,
            maxNoncesPerRound: 1 << 18)

        let result = try MultiRoundGrinding.grindAllRounds(
            roundSeeds: roundSeeds, config: config, engine: engine)

        // The second round's effective seed should incorporate the first nonce
        let expectedSeed1 = poseidon2Hash(roundSeeds[1], frFromInt(result.nonces[0]))
        let hash1 = GrindingVerifier.computeHash(
            seed: expectedSeed1, nonce: result.nonces[1],
            hashFunction: .poseidon2)
        let lz = GrindingVerifier.countLeadingZeros(hash1)
        expect(lz >= 3, "Chained round should meet difficulty, got \(lz) leading zeros")

    } catch {
        expect(false, "Multi-round chaining test failed: \(error)")
    }
}

// MARK: - Grinding Config Tests

private func testGrindingConfigDefaults() {
    suite("FRI Grinding: Config defaults")

    let config = GrindingConfig()

    expectEqual(config.difficulty, 16, "Default difficulty should be 16")
    expectEqual(config.batchSize, 1 << 20, "Default batch size should be 1M")
    expect(config.cpuVerify, "CPU verify should be on by default")
    expect(config.maxNonces == (1 << 32), "Default maxNonces should be 2^32")
}

private func testGrindingConfigCustom() {
    suite("FRI Grinding: Custom config")

    let config = GrindingConfig(difficulty: 20,
                                hashFunction: .keccak256,
                                maxNonces: 1_000_000,
                                batchSize: 1 << 16,
                                cpuVerify: false)

    expectEqual(config.difficulty, 20, "Custom difficulty should be 20")
    expectEqual(config.batchSize, 1 << 16, "Custom batch size should be 64K")
    expect(!config.cpuVerify, "CPU verify should be off")
    expectEqual(config.maxNonces, 1_000_000, "Custom maxNonces")
}

// MARK: - Statistics Tests

private func testGrindingStatistics() {
    suite("FRI Grinding: Statistics tracking")

    do {
        let engine = try GPUFRIGrindingEngine()

        engine.resetStatistics()
        expectEqual(engine.statistics.grindCount, 0, "Initial grind count should be 0")
        expectEqual(engine.statistics.totalNoncesChecked, 0, "Initial nonces should be 0")

        // Perform a grind
        let seed = frFromInt(42)
        let config = GrindingConfig(difficulty: 3, hashFunction: .poseidon2,
                                    maxNonces: 1 << 16, batchSize: 1 << 12)
        _ = try engine.grind(seed: seed, config: config)

        expectEqual(engine.statistics.grindCount, 1, "After one grind, count should be 1")
        expectEqual(engine.statistics.successCount, 1, "Should have 1 success")
        expectEqual(engine.statistics.failureCount, 0, "Should have 0 failures")
        expect(engine.statistics.totalNoncesChecked > 0, "Should have checked nonces")
        expect(engine.statistics.averageHashRate > 0, "Average hash rate should be positive")

        // Perform another grind
        _ = try engine.grind(seed: frFromInt(99), config: config)
        expectEqual(engine.statistics.grindCount, 2, "After two grinds, count should be 2")
        expectEqual(engine.statistics.successCount, 2, "Should have 2 successes")

        // Reset
        engine.resetStatistics()
        expectEqual(engine.statistics.grindCount, 0, "After reset, count should be 0")
        expectEqual(engine.statistics.totalNoncesChecked, 0, "After reset, nonces should be 0")

    } catch {
        expect(false, "Statistics test failed: \(error)")
    }
}

// MARK: - Transcript Helper Tests

private func testTranscriptHelper() {
    suite("FRI Grinding: Transcript helper")

    var rng = GrindRNG(state: 888)
    var elements = [Fr]()
    for _ in 0..<4 { elements.append(rng.nextFr()) }

    let seed = GrindingTranscriptHelper.deriveGrindingSeed(transcriptElements: elements)
    expect(!seed.isZero, "Derived seed should not be zero")

    let seed2 = GrindingTranscriptHelper.deriveGrindingSeed(transcriptElements: elements)
    expect(frEqual(seed, seed2), "Same elements should produce same seed")

    var altElements = elements
    altElements[2] = rng.nextFr()
    let seed3 = GrindingTranscriptHelper.deriveGrindingSeed(transcriptElements: altElements)
    expect(!frEqual(seed, seed3), "Different elements should produce different seed")

    let seedEmpty = GrindingTranscriptHelper.deriveGrindingSeed(transcriptElements: [])
    expect(seedEmpty.isZero, "Empty elements should produce zero seed")
}

private func testPostGrindingState() {
    suite("FRI Grinding: Post-grinding state")

    let preState = frFromInt(12345)
    let nonce: UInt64 = 67890

    let postState = GrindingTranscriptHelper.postGrindingState(
        preGrindState: preState, nonce: nonce)
    expect(!postState.isZero, "Post state should not be zero")

    // Deterministic
    let postState2 = GrindingTranscriptHelper.postGrindingState(
        preGrindState: preState, nonce: nonce)
    expect(frEqual(postState, postState2), "Post state should be deterministic")

    // Should equal poseidon2Hash(preState, frFromInt(nonce))
    let expected = poseidon2Hash(preState, frFromInt(nonce))
    expect(frEqual(postState, expected),
           "Post state should equal poseidon2Hash(preState, nonce)")
}

// MARK: - Adaptive Difficulty Tests

private func testAdaptiveDifficulty() {
    suite("FRI Grinding: Adaptive difficulty controller")

    var controller = AdaptiveDifficultyController(
        targetTimeSeconds: 1.0, minDifficulty: 8, maxDifficulty: 24,
        smoothingFactor: 0.5, initialHashRate: 1_000_000)

    let initial = controller.recommendedDifficulty
    expectEqual(initial, 19, "log2(1M) = ~19")

    // Fast result should increase difficulty
    let fastResult = GrindingResult(
        nonce: 42, hashOutput: Fr.zero, noncesChecked: 10_000_000,
        elapsedSeconds: 0.1, cpuVerified: true)
    controller.update(from: fastResult)
    let afterFast = controller.recommendedDifficulty
    expect(afterFast >= initial, "Faster hash rate should increase difficulty")

    // Slow result should decrease
    let slowResult = GrindingResult(
        nonce: 99, hashOutput: Fr.zero, noncesChecked: 100,
        elapsedSeconds: 1.0, cpuVerified: true)
    controller.update(from: slowResult)
    expect(controller.recommendedDifficulty <= afterFast,
           "Slower hash rate should decrease difficulty")

    // Min/max clamping
    let fast = AdaptiveDifficultyController(
        targetTimeSeconds: 1.0, minDifficulty: 12, maxDifficulty: 16,
        initialHashRate: 1_000_000_000_000)
    expect(fast.recommendedDifficulty <= 16, "Should clamp to max 16")

    let slow = AdaptiveDifficultyController(
        targetTimeSeconds: 1.0, minDifficulty: 12, maxDifficulty: 16,
        initialHashRate: 1)
    expect(slow.recommendedDifficulty >= 12, "Should clamp to min 12")
}

// MARK: - Time Estimation Tests

private func testTimeEstimation() {
    suite("FRI Grinding: Time estimation")

    do {
        let engine = try GPUFRIGrindingEngine()

        let timePoseidon = engine.estimateTime(difficulty: 20, hashFunction: .poseidon2)
        let timeKeccak = engine.estimateTime(difficulty: 20, hashFunction: .keccak256)
        expect(timePoseidon > 0, "Estimated time should be positive")
        expect(timeKeccak > timePoseidon, "Keccak should be slower than Poseidon2")

        let time10 = engine.estimateTime(difficulty: 10, hashFunction: .poseidon2)
        let time20 = engine.estimateTime(difficulty: 20, hashFunction: .poseidon2)
        let ratio = time20 / time10
        expect(ratio > 900 && ratio < 1200,
               "Difficulty 20 should be ~1024x slower than 10, got \(ratio)")
    } catch {
        expect(false, "Time estimation test failed: \(error)")
    }
}

// MARK: - Edge Case Tests

private func testZeroSeedGrinding() {
    suite("FRI Grinding: Zero seed grinding")

    do {
        let engine = try GPUFRIGrindingEngine()
        let config = GrindingConfig(difficulty: 3, hashFunction: .poseidon2,
                                    maxNonces: 1 << 16, batchSize: 1 << 12,
                                    cpuVerify: true)

        let result = try engine.grind(seed: Fr.zero, config: config)
        let hash = result.hashOutput
        let lz = GrindingVerifier.countLeadingZeros(hash)
        expect(lz >= 3, "Should find valid nonce even with zero seed")

    } catch {
        expect(false, "Zero seed grinding failed: \(error)")
    }
}

private func testOneSeedGrinding() {
    suite("FRI Grinding: Fr.one seed grinding")

    do {
        let engine = try GPUFRIGrindingEngine()
        let config = GrindingConfig(difficulty: 4, hashFunction: .poseidon2,
                                    maxNonces: 1 << 18, batchSize: 1 << 14,
                                    cpuVerify: true)

        let result = try engine.grind(seed: Fr.one, config: config)
        expect(result.cpuVerified, "Result should be CPU verified")

        let proof = GrindingProof(seed: Fr.one, nonce: result.nonce,
                                   difficulty: 4, hashFunction: .poseidon2)
        expect(proof.verify(), "Proof from Fr.one seed should verify")

    } catch {
        expect(false, "Fr.one seed grinding failed: \(error)")
    }
}

private func testMinDifficultyGrinding() {
    suite("FRI Grinding: Minimum difficulty (1 bit)")

    do {
        let engine = try GPUFRIGrindingEngine()
        let config = GrindingConfig(difficulty: 1, hashFunction: .poseidon2,
                                    maxNonces: 1 << 10, batchSize: 1 << 8,
                                    cpuVerify: true)

        let result = try engine.grind(seed: frFromInt(42), config: config)
        // With difficulty 1, roughly 50% of nonces should work
        // So we should find one very quickly
        expect(result.noncesChecked <= 10,
               "Difficulty 1 should find nonce very quickly, took \(result.noncesChecked)")

    } catch {
        expect(false, "Min difficulty grinding failed: \(error)")
    }
}

private func testGrindMultipleSeeds() {
    suite("FRI Grinding: Grind multiple seeds")

    do {
        let engine = try GPUFRIGrindingEngine()
        var rng = GrindRNG(state: 666)

        var seeds = [Fr]()
        for _ in 0..<4 {
            seeds.append(rng.nextFr())
        }

        let config = GrindingConfig(difficulty: 3, hashFunction: .poseidon2,
                                    maxNonces: 1 << 16, batchSize: 1 << 12,
                                    cpuVerify: true)

        let results = try engine.grindMultiple(seeds: seeds, config: config)
        expectEqual(results.count, 4, "Should have 4 results")

        for (i, result) in results.enumerated() {
            let hash = GrindingVerifier.computeHash(
                seed: seeds[i], nonce: result.nonce,
                hashFunction: .poseidon2)
            let lz = GrindingVerifier.countLeadingZeros(hash)
            expect(lz >= 3, "Result \(i) should meet difficulty 3, got \(lz)")
        }

    } catch {
        expect(false, "Grind multiple seeds failed: \(error)")
    }
}

// MARK: - Keccak Hash Consistency Tests

private func testKeccakHashConsistency() {
    suite("FRI Grinding: Keccak hash consistency")

    let seed = frFromInt(42)
    let nonce: UInt64 = 12345

    // Compute hash twice and verify same result
    let hash1 = GrindingVerifier.computeHash(seed: seed, nonce: nonce,
                                              hashFunction: .keccak256)
    let hash2 = GrindingVerifier.computeHash(seed: seed, nonce: nonce,
                                              hashFunction: .keccak256)
    expect(frEqual(hash1, hash2), "Same seed+nonce should produce same Keccak hash")

    // Different nonce should produce different hash
    let hash3 = GrindingVerifier.computeHash(seed: seed, nonce: nonce + 1,
                                              hashFunction: .keccak256)
    expect(!frEqual(hash1, hash3), "Different nonce should produce different hash")

    // Poseidon2 and Keccak should differ
    let hashPos = GrindingVerifier.computeHash(seed: seed, nonce: nonce,
                                                hashFunction: .poseidon2)
    expect(!frEqual(hash1, hashPos), "Keccak and Poseidon2 should differ")
}

// MARK: - Poseidon2 Hash Consistency Tests

private func testPoseidon2HashConsistency() {
    suite("FRI Grinding: Poseidon2 hash consistency")

    let seed = frFromInt(0xABCD)
    let nonce: UInt64 = 9999

    // computeHash should match direct poseidon2Hash call
    let computed = GrindingVerifier.computeHash(
        seed: seed, nonce: nonce, hashFunction: .poseidon2)
    let direct = poseidon2Hash(seed, frFromInt(nonce))
    expect(frEqual(computed, direct),
           "computeHash(.poseidon2) should match poseidon2Hash")
}

// MARK: - Grinding Proof Field Tests

private func testGrindingProofFields() {
    suite("FRI Grinding: Proof field integrity")

    let seed = frFromInt(0x1234)
    let nonce: UInt64 = 56789

    let proofP2 = GrindingProof(seed: seed, nonce: nonce, difficulty: 12,
                                 hashFunction: .poseidon2)
    expect(frEqual(proofP2.seed, seed), "Proof seed should match")
    expectEqual(proofP2.nonce, nonce, "Proof nonce should match")
    expectEqual(proofP2.difficulty, 12, "Proof difficulty should match")

    let proofKec = GrindingProof(seed: seed, nonce: nonce, difficulty: 12,
                                  hashFunction: .keccak256)
    expect(!frEqual(proofP2.computeHash(), proofKec.computeHash()),
           "Different hash functions should produce different proof hashes")
}

// MARK: - CPU Fallback Tests

private func testCPUFallbackThreshold() {
    suite("FRI Grinding: CPU fallback for low difficulty")

    do {
        let engine = try GPUFRIGrindingEngine()

        // Difficulty below threshold should use CPU path
        let config = GrindingConfig(
            difficulty: GPUFRIGrindingEngine.cpuFallbackDifficultyThreshold - 1,
            hashFunction: .poseidon2,
            maxNonces: 1 << 16, batchSize: 1 << 12,
            cpuVerify: true)

        let result = try engine.grind(seed: frFromInt(42), config: config)
        expect(result.cpuVerified, "Low difficulty should be CPU verified")

        let hash = result.hashOutput
        let lz = GrindingVerifier.countLeadingZeros(hash)
        let threshold = GPUFRIGrindingEngine.cpuFallbackDifficultyThreshold - 1
        expect(lz >= threshold,
               "Should meet target difficulty \(threshold), got \(lz)")

    } catch {
        expect(false, "CPU fallback test failed: \(error)")
    }
}

// MARK: - Commit Phase Grinding Flow Tests

private func testCommitPhaseGrindingFlow() {
    suite("FRI Grinding: Full commit phase grinding flow")

    do {
        let engine = try GPUFRIGrindingEngine()
        var rng = GrindRNG(state: 1234)

        // Simulate FRI commit phase data
        var commitRoots = [Fr]()
        var challenges = [Fr]()
        for _ in 0..<4 {
            commitRoots.append(rng.nextFr())
            challenges.append(rng.nextFr())
        }

        let (nonce, postState, result) = try GrindingTranscriptHelper.commitPhaseGrinding(
            commitRoots: commitRoots, challenges: challenges,
            difficulty: 4, engine: engine)

        // Nonce should be valid
        expect(result.noncesChecked > 0, "Should have checked nonces")

        // Post state should be derived from seed + nonce
        let seed = FRIGrindingIntegration.computeGrindingSeed(
            commitRoots: commitRoots, foldingChallenges: challenges)
        let expectedPost = GrindingTranscriptHelper.postGrindingState(
            preGrindState: seed, nonce: nonce)
        expect(frEqual(postState, expectedPost),
               "Post state should match expected derivation")

        // Verify the proof manually
        let hash = GrindingVerifier.computeHash(
            seed: seed, nonce: nonce, hashFunction: .poseidon2)
        let lz = GrindingVerifier.countLeadingZeros(hash)
        expect(lz >= 4, "Grinding hash should meet difficulty 4, got \(lz)")

    } catch {
        expect(false, "Commit phase grinding flow failed: \(error)")
    }
}

// MARK: - Higher Difficulty Tests

private func testModerateDifficultyGrinding() {
    suite("FRI Grinding: Moderate difficulty (8 bits)")

    do {
        let engine = try GPUFRIGrindingEngine()
        let config = GrindingConfig(difficulty: 8, hashFunction: .poseidon2,
                                    maxNonces: 1 << 20, batchSize: 1 << 16,
                                    cpuVerify: true)

        let result = try engine.grind(seed: frFromInt(0xFACE), config: config)
        let lz = GrindingVerifier.countLeadingZeros(result.hashOutput)
        expect(lz >= 8, "Should meet difficulty 8, got \(lz) leading zeros")
        expect(result.noncesChecked < 10000,
               "Should find 8-bit target quickly, took \(result.noncesChecked)")
    } catch {
        expect(false, "Moderate difficulty test failed: \(error)")
    }
}

// MARK: - Multi-Round Verification Edge Cases

private func testMultiRoundVerificationMismatch() {
    suite("FRI Grinding: Multi-round verification with wrong data")

    do {
        let engine = try GPUFRIGrindingEngine()
        var rng = GrindRNG(state: 999)
        var roundSeeds = [Fr]()
        for _ in 0..<2 { roundSeeds.append(rng.nextFr()) }

        let config = MultiRoundGrinding.MultiRoundConfig(
            perRoundDifficulty: [3], hashFunction: .poseidon2,
            maxNoncesPerRound: 1 << 18)

        let result = try MultiRoundGrinding.grindAllRounds(
            roundSeeds: roundSeeds, config: config, engine: engine)

        expect(MultiRoundGrinding.verifyMultiRound(
            roundSeeds: roundSeeds, result: result, config: config),
               "Correct multi-round should verify")

        var wrongSeeds = roundSeeds
        wrongSeeds[0] = rng.nextFr()
        expect(!MultiRoundGrinding.verifyMultiRound(
            roundSeeds: wrongSeeds, result: result, config: config),
               "Wrong seeds should fail verification")

        expect(!MultiRoundGrinding.verifyMultiRound(
            roundSeeds: [roundSeeds[0]], result: result, config: config),
               "Mismatched seed count should fail")
    } catch {
        expect(false, "Multi-round verification mismatch test failed: \(error)")
    }
}

// MARK: - Public Test Runner

public func runGPUFRIGrindingTests() {
    // Leading zeros and target checks
    testLeadingZeroCount()
    testMeetsTarget()

    // Poseidon2 grinding
    testPoseidon2Grinding()
    testPoseidon2GrindingDeterminism()
    testPoseidon2GrindingDifferentSeeds()

    // Keccak grinding
    testKeccakGrinding()
    testKeccakVsPoseidon2()

    // Grinding proofs
    testGrindingProof()
    testInvalidProof()

    // Batch verification
    testBatchVerification()
    testBatchVerificationWithInvalid()

    // FRI integration
    testGrindingSeedComputation()
    testQueryDerivation()

    // Difficulty estimation
    testDifficultyEstimator()
    testDifficultyRecommendation()

    // Nonce strategies
    testLinearNonceGenerator()
    testStridedNonceGenerator()
    testRandomNonceGenerator()

    // Multi-round grinding
    testMultiRoundGrinding()
    testMultiRoundChaining()

    // Config
    testGrindingConfigDefaults()
    testGrindingConfigCustom()

    // Statistics
    testGrindingStatistics()

    // Transcript helpers
    testTranscriptHelper()
    testPostGrindingState()

    // Adaptive difficulty
    testAdaptiveDifficulty()

    // Time estimation
    testTimeEstimation()

    // Edge cases
    testZeroSeedGrinding()
    testOneSeedGrinding()
    testMinDifficultyGrinding()
    testGrindMultipleSeeds()

    // Hash consistency
    testKeccakHashConsistency()
    testPoseidon2HashConsistency()

    // Proof fields
    testGrindingProofFields()

    // CPU fallback
    testCPUFallbackThreshold()

    // Integration flows
    testCommitPhaseGrindingFlow()
    testModerateDifficultyGrinding()

    // Multi-round edge cases
    testMultiRoundVerificationMismatch()
}
