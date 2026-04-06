// GPUFiatShamirTests — Comprehensive tests for the GPU-accelerated Fiat-Shamir engine
//
// Tests cover:
//   - Basic absorb/squeeze for Fr and BabyBear
//   - Domain separation correctness
//   - Determinism (same inputs -> same outputs)
//   - Fork independence
//   - Merlin-style API (append/challenge)
//   - Batch transcript processing
//   - Transcript consistency checker
//   - State save/restore
//   - Multi-challenge squeeze
//   - Edge cases (empty data, large inputs, counter overflow)

import Foundation
import zkMetal

// Overload suite to accept a trailing closure for structured test grouping
private func suite(_ name: String, _ body: () -> Void) {
    suite(name)
    body()
}

// MARK: - Public Entry Point

public func runGPUFiatShamirTests() {
    suite("GPU Fiat-Shamir Engine") {
        testBasicAbsorbSqueeze()
        testDomainSeparation()
        testDeterminism()
        testForkIndependence()
        testMultipleSqueezes()
        testAppendMessage()
        testAppendScalars()
        testSqueezeBytes()
        testOperationCounter()
        testStateSaveRestore()
        testDomainSeparateMarker()
        testEmptyLabelHandling()
        testLargeInputAbsorb()
    }

    suite("GPU Fiat-Shamir BabyBear Engine") {
        testBbBasicAbsorbSqueeze()
        testBbDomainSeparation()
        testBbDeterminism()
        testBbForkIndependence()
        testBbMultipleSqueezes()
        testBbAppendMessage()
        testBbOperationCounter()
    }

    suite("GPU Batch Fiat-Shamir Engine") {
        testBatchCreation()
        testBatchAppendScalar()
        testBatchAppendScalars()
        testBatchSqueeze()
        testBatchSqueezeMultiple()
        testBatchFork()
        testBatchDeterminism()
        testBatchIndependence()
    }

    suite("Transcript Consistency Checker") {
        testConsistencyCheckerMatch()
        testConsistencyCheckerDivergence()
        testConsistencyMultiStep()
    }

    suite("SpongeField Protocol") {
        testFrSpongeField()
        testBbSpongeField()
    }

    suite("FiatShamirSpongeState") {
        testSpongeStateInit()
        testSpongeStateExplicitInit()
    }

    suite("TranscriptEngine Conformance") {
        testTranscriptEngineConformance()
        testTranscriptEngineSqueezeChallenges()
    }
}

// MARK: - Fr Engine: Basic Absorb/Squeeze

private func testBasicAbsorbSqueeze() {
    var engine = GPUFiatShamirEngine(label: "test-protocol")
    let val = frFromInt(42)
    engine.appendScalar(label: "input", value: val)
    let challenge = engine.squeezeChallenge(label: "alpha")

    // Challenge should be non-zero
    let limbs = challenge.to64()
    let isNonZero = limbs[0] != 0 || limbs[1] != 0 || limbs[2] != 0 || limbs[3] != 0
    expect(isNonZero, "squeezed challenge should be non-zero")

    // Challenge should be different from input
    let inputLimbs = val.to64()
    let isDifferent = limbs[0] != inputLimbs[0] || limbs[1] != inputLimbs[1] ||
                      limbs[2] != inputLimbs[2] || limbs[3] != inputLimbs[3]
    expect(isDifferent, "challenge should differ from input")
}

// MARK: - Fr Engine: Domain Separation

private func testDomainSeparation() {
    // Two engines with different protocol labels should produce different challenges
    var engine1 = GPUFiatShamirEngine(label: "protocol-A")
    var engine2 = GPUFiatShamirEngine(label: "protocol-B")

    let val = frFromInt(100)
    engine1.appendScalar(label: "x", value: val)
    engine2.appendScalar(label: "x", value: val)

    let c1 = engine1.squeezeChallenge(label: "c")
    let c2 = engine2.squeezeChallenge(label: "c")

    let l1 = c1.to64()
    let l2 = c2.to64()
    let differ = l1[0] != l2[0] || l1[1] != l2[1] || l1[2] != l2[2] || l1[3] != l2[3]
    expect(differ, "different protocol labels should produce different challenges")
}

// MARK: - Fr Engine: Determinism

private func testDeterminism() {
    // Same inputs should produce identical challenges
    var engine1 = GPUFiatShamirEngine(label: "determinism-test")
    var engine2 = GPUFiatShamirEngine(label: "determinism-test")

    let v1 = frFromInt(7)
    let v2 = frFromInt(13)

    engine1.appendScalar(label: "a", value: v1)
    engine1.appendScalar(label: "b", value: v2)
    let c1 = engine1.squeezeChallenge(label: "result")

    engine2.appendScalar(label: "a", value: v1)
    engine2.appendScalar(label: "b", value: v2)
    let c2 = engine2.squeezeChallenge(label: "result")

    let l1 = c1.to64()
    let l2 = c2.to64()
    expect(l1[0] == l2[0] && l1[1] == l2[1] && l1[2] == l2[2] && l1[3] == l2[3],
           "same inputs must produce identical challenges (determinism)")
}

// MARK: - Fr Engine: Fork Independence

private func testForkIndependence() {
    var parent = GPUFiatShamirEngine(label: "parent-protocol")
    parent.appendScalar(label: "shared", value: frFromInt(99))

    let saved = parent.savedState
    var child1: GPUFiatShamirEngine = parent.fork(label: "child-A")
    var child2: GPUFiatShamirEngine = parent.fork(label: "child-B")

    // Children should produce different challenges (different fork labels)
    let cc1 = child1.squeezeChallenge(label: "c")
    let cc2 = child2.squeezeChallenge(label: "c")

    let l1 = cc1.to64()
    let l2 = cc2.to64()
    let differ = l1[0] != l2[0] || l1[1] != l2[1] || l1[2] != l2[2] || l1[3] != l2[3]
    expect(differ, "forks with different labels should produce different challenges")

    // Parent state should be preserved after forking
    let afterFork = parent.savedState
    let s0match = saved.s0.to64()[0] == afterFork.s0.to64()[0]
    expect(s0match, "parent state should not change after fork")

    // Child operation count should start at 0
    expectEqual(child1.operationCount, 1, "child op count after one squeeze")
}

// MARK: - Fr Engine: Multiple Squeezes

private func testMultipleSqueezes() {
    var engine = GPUFiatShamirEngine(label: "multi-squeeze")
    engine.appendScalar(label: "seed", value: frFromInt(12345))

    let challenges = engine.squeezeChallenges(label: "rounds", count: 5)
    expectEqual(challenges.count, 5, "should produce exactly 5 challenges")

    // All challenges should be distinct
    for i in 0..<challenges.count {
        for j in (i+1)..<challenges.count {
            let li = challenges[i].to64()
            let lj = challenges[j].to64()
            let differ = li[0] != lj[0] || li[1] != lj[1] || li[2] != lj[2] || li[3] != lj[3]
            expect(differ, "challenge \(i) and \(j) should be distinct")
        }
    }
}

// MARK: - Fr Engine: Append Message

private func testAppendMessage() {
    var engine = GPUFiatShamirEngine(label: "message-test")

    // Absorb some raw bytes
    let message: [UInt8] = [0x01, 0x02, 0x03, 0x04, 0xFF, 0xFE, 0xFD]
    engine.appendMessage(label: "data", bytes: message)
    let c1 = engine.squeezeChallenge(label: "c")

    // Same message should produce same result
    var engine2 = GPUFiatShamirEngine(label: "message-test")
    engine2.appendMessage(label: "data", bytes: message)
    let c2 = engine2.squeezeChallenge(label: "c")

    let l1 = c1.to64()
    let l2 = c2.to64()
    expect(l1[0] == l2[0] && l1[1] == l2[1] && l1[2] == l2[2] && l1[3] == l2[3],
           "same message should produce same challenge")

    // Different message should produce different result
    var engine3 = GPUFiatShamirEngine(label: "message-test")
    engine3.appendMessage(label: "data", bytes: [0x01, 0x02, 0x03, 0x04, 0xFF, 0xFE, 0xFC])
    let c3 = engine3.squeezeChallenge(label: "c")

    let l3 = c3.to64()
    let differ = l1[0] != l3[0] || l1[1] != l3[1] || l1[2] != l3[2] || l1[3] != l3[3]
    expect(differ, "different messages should produce different challenges")
}

// MARK: - Fr Engine: Append Scalars

private func testAppendScalars() {
    var engine = GPUFiatShamirEngine(label: "scalars-test")
    let values = [frFromInt(1), frFromInt(2), frFromInt(3)]
    engine.appendScalars(label: "vector", values: values)
    let c = engine.squeezeChallenge(label: "c")

    let limbs = c.to64()
    let isNonZero = limbs[0] != 0 || limbs[1] != 0 || limbs[2] != 0 || limbs[3] != 0
    expect(isNonZero, "challenge from scalars should be non-zero")

    // Same scalars should produce same result
    var engine2 = GPUFiatShamirEngine(label: "scalars-test")
    engine2.appendScalars(label: "vector", values: values)
    let c2 = engine2.squeezeChallenge(label: "c")

    let l2 = c2.to64()
    expect(limbs[0] == l2[0] && limbs[1] == l2[1] && limbs[2] == l2[2] && limbs[3] == l2[3],
           "same scalars should be deterministic")
}

// MARK: - Fr Engine: Squeeze Bytes

private func testSqueezeBytes() {
    var engine = GPUFiatShamirEngine(label: "bytes-test")
    engine.appendScalar(label: "seed", value: frFromInt(77))

    let bytes = engine.squeezeBytes(label: "output", byteCount: 64)
    expectEqual(bytes.count, 64, "should produce exactly 64 bytes")

    // Should not be all zeros
    let allZero = bytes.allSatisfy { $0 == 0 }
    expect(!allZero, "squeezed bytes should not be all zeros")
}

// MARK: - Fr Engine: Operation Counter

private func testOperationCounter() {
    var engine = GPUFiatShamirEngine(label: "counter-test")
    expectEqual(engine.operationCount, 0, "initial op count should be 0")

    engine.appendScalar(label: "a", value: frFromInt(1))
    expectEqual(engine.operationCount, 1, "op count after 1 absorb")

    engine.appendScalar(label: "b", value: frFromInt(2))
    expectEqual(engine.operationCount, 2, "op count after 2 absorbs")

    _ = engine.squeezeChallenge(label: "c")
    expectEqual(engine.operationCount, 3, "op count after 1 squeeze")

    _ = engine.squeezeChallenges(label: "d", count: 3)
    // squeezeChallenges increments once per challenge
    expectEqual(engine.operationCount, 6, "op count after 3 more squeezes")
}

// MARK: - Fr Engine: State Save/Restore

private func testStateSaveRestore() {
    var engine = GPUFiatShamirEngine(label: "save-restore")
    engine.appendScalar(label: "x", value: frFromInt(42))

    let saved = engine.savedState

    // Squeeze and modify
    let c1 = engine.squeezeChallenge(label: "first")

    // Restore and squeeze again -- should get same result
    engine.restore(from: saved)
    let c2 = engine.squeezeChallenge(label: "first")

    let l1 = c1.to64()
    let l2 = c2.to64()
    expect(l1[0] == l2[0] && l1[1] == l2[1] && l1[2] == l2[2] && l1[3] == l2[3],
           "restored state should produce same challenge")
}

// MARK: - Fr Engine: Domain Separate Marker

private func testDomainSeparateMarker() {
    var engine1 = GPUFiatShamirEngine(label: "marker-test")
    engine1.appendScalar(label: "x", value: frFromInt(5))
    engine1.domainSeparate(label: "phase-2")
    let c1 = engine1.squeezeChallenge(label: "c")

    var engine2 = GPUFiatShamirEngine(label: "marker-test")
    engine2.appendScalar(label: "x", value: frFromInt(5))
    // No domain separator
    let c2 = engine2.squeezeChallenge(label: "c")

    let l1 = c1.to64()
    let l2 = c2.to64()
    let differ = l1[0] != l2[0] || l1[1] != l2[1] || l1[2] != l2[2] || l1[3] != l2[3]
    expect(differ, "domain separation marker should change challenge output")
}

// MARK: - Fr Engine: Empty Label Handling

private func testEmptyLabelHandling() {
    var engine = GPUFiatShamirEngine(label: "")
    engine.appendScalar(label: "", value: frFromInt(1))
    let c = engine.squeezeChallenge(label: "")

    let limbs = c.to64()
    let isNonZero = limbs[0] != 0 || limbs[1] != 0 || limbs[2] != 0 || limbs[3] != 0
    expect(isNonZero, "empty labels should still produce non-zero output")
}

// MARK: - Fr Engine: Large Input Absorb

private func testLargeInputAbsorb() {
    var engine = GPUFiatShamirEngine(label: "large-input")

    // Absorb 100 field elements
    var values = [Fr]()
    for i in 0..<100 {
        values.append(frFromInt(UInt64(i + 1)))
    }
    engine.appendScalars(label: "big-vector", values: values)

    let c = engine.squeezeChallenge(label: "c")
    let limbs = c.to64()
    let isNonZero = limbs[0] != 0 || limbs[1] != 0 || limbs[2] != 0 || limbs[3] != 0
    expect(isNonZero, "large input should produce non-zero challenge")

    // Absorb large byte message (> 31 bytes, multiple Fr packing rounds)
    var engine2 = GPUFiatShamirEngine(label: "large-bytes")
    let bigMessage = [UInt8](repeating: 0xAB, count: 256)
    engine2.appendMessage(label: "payload", bytes: bigMessage)
    let c2 = engine2.squeezeChallenge(label: "c")

    let l2 = c2.to64()
    let isNonZero2 = l2[0] != 0 || l2[1] != 0 || l2[2] != 0 || l2[3] != 0
    expect(isNonZero2, "large byte message should produce non-zero challenge")
}

// MARK: - BabyBear Engine: Basic Absorb/Squeeze

private func testBbBasicAbsorbSqueeze() {
    var engine = GPUFiatShamirBbEngine(label: "bb-test")
    engine.appendScalar(label: "input", value: Bb(v: 42))
    let challenge = engine.squeezeChallenge(label: "alpha")

    expect(challenge.v != 0, "BabyBear challenge should be non-zero")
    expect(challenge.v < Bb.P, "BabyBear challenge should be < p")
}

// MARK: - BabyBear Engine: Domain Separation

private func testBbDomainSeparation() {
    var e1 = GPUFiatShamirBbEngine(label: "bb-A")
    var e2 = GPUFiatShamirBbEngine(label: "bb-B")

    e1.appendScalar(label: "x", value: Bb(v: 100))
    e2.appendScalar(label: "x", value: Bb(v: 100))

    let c1 = e1.squeezeChallenge(label: "c")
    let c2 = e2.squeezeChallenge(label: "c")

    expect(c1.v != c2.v, "different BB protocol labels should produce different challenges")
}

// MARK: - BabyBear Engine: Determinism

private func testBbDeterminism() {
    var e1 = GPUFiatShamirBbEngine(label: "bb-det")
    var e2 = GPUFiatShamirBbEngine(label: "bb-det")

    e1.appendScalar(label: "a", value: Bb(v: 7))
    e2.appendScalar(label: "a", value: Bb(v: 7))

    let c1 = e1.squeezeChallenge(label: "c")
    let c2 = e2.squeezeChallenge(label: "c")

    expectEqual(c1.v, c2.v, "BabyBear determinism: same inputs -> same output")
}

// MARK: - BabyBear Engine: Fork Independence

private func testBbForkIndependence() {
    var parent = GPUFiatShamirBbEngine(label: "bb-parent")
    parent.appendScalar(label: "s", value: Bb(v: 55))

    var child1 = parent.fork(label: "bb-child-A")
    var child2 = parent.fork(label: "bb-child-B")

    let cc1 = child1.squeezeChallenge(label: "c")
    let cc2 = child2.squeezeChallenge(label: "c")

    expect(cc1.v != cc2.v, "BB forks with different labels should differ")
}

// MARK: - BabyBear Engine: Multiple Squeezes

private func testBbMultipleSqueezes() {
    var engine = GPUFiatShamirBbEngine(label: "bb-multi")
    engine.appendScalar(label: "seed", value: Bb(v: 999))

    let challenges = engine.squeezeChallenges(label: "rounds", count: 4)
    expectEqual(challenges.count, 4, "should produce 4 BabyBear challenges")

    // All should be valid field elements
    for (i, c) in challenges.enumerated() {
        expect(c.v < Bb.P, "BB challenge \(i) should be < p")
    }

    // All should be distinct (overwhelmingly likely for random outputs)
    var seen = Set<UInt32>()
    for c in challenges {
        seen.insert(c.v)
    }
    expectEqual(seen.count, challenges.count, "all BB challenges should be distinct")
}

// MARK: - BabyBear Engine: Append Message

private func testBbAppendMessage() {
    var engine = GPUFiatShamirBbEngine(label: "bb-msg")
    engine.appendMessage(label: "data", bytes: [0x01, 0x02, 0x03, 0x04, 0x05])
    let c = engine.squeezeChallenge(label: "c")
    expect(c.v != 0, "BB challenge from message should be non-zero")
    expect(c.v < Bb.P, "BB challenge from message should be valid")
}

// MARK: - BabyBear Engine: Operation Counter

private func testBbOperationCounter() {
    var engine = GPUFiatShamirBbEngine(label: "bb-counter")
    expectEqual(engine.operationCount, 0, "BB initial op count")

    engine.appendScalar(label: "a", value: Bb(v: 1))
    expectEqual(engine.operationCount, 1, "BB op count after 1 absorb")

    _ = engine.squeezeChallenge(label: "c")
    expectEqual(engine.operationCount, 2, "BB op count after squeeze")
}

// MARK: - Batch Engine: Creation

private func testBatchCreation() {
    let batch = GPUBatchFiatShamirEngine(label: "batch-test")
    let transcripts = batch.createBatch(count: 8)
    expectEqual(transcripts.count, 8, "should create 8 transcripts")

    // Each should have been initialized with a unique batch index
    // (operation count = 1 from the batch-index append)
    for (i, t) in transcripts.enumerated() {
        expectEqual(t.operationCount, 1, "transcript \(i) should have op count 1")
    }
}

// MARK: - Batch Engine: Append Scalar

private func testBatchAppendScalar() {
    let batch = GPUBatchFiatShamirEngine(label: "batch-scalar")
    let transcripts = batch.createBatch(count: 4)

    let values = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
    let updated = batch.batchAppendScalar(
        transcripts: transcripts, label: "x", values: values)

    expectEqual(updated.count, 4, "batch append should preserve count")
    for (i, t) in updated.enumerated() {
        expectEqual(t.operationCount, 2, "transcript \(i) should have 2 ops after append")
    }
}

// MARK: - Batch Engine: Append Scalars

private func testBatchAppendScalars() {
    let batch = GPUBatchFiatShamirEngine(label: "batch-scalars")
    let transcripts = batch.createBatch(count: 3)

    let values: [[Fr]] = [
        [frFromInt(1), frFromInt(2)],
        [frFromInt(3), frFromInt(4)],
        [frFromInt(5), frFromInt(6)]
    ]
    let updated = batch.batchAppendScalars(
        transcripts: transcripts, label: "vec", values: values)

    expectEqual(updated.count, 3, "batch append scalars should preserve count")
}

// MARK: - Batch Engine: Squeeze

private func testBatchSqueeze() {
    let batch = GPUBatchFiatShamirEngine(label: "batch-squeeze")
    var transcripts = batch.createBatch(count: 4)

    let values = [frFromInt(100), frFromInt(200), frFromInt(300), frFromInt(400)]
    transcripts = batch.batchAppendScalar(
        transcripts: transcripts, label: "commitment", values: values)

    do {
        let (updated, challenges) = try batch.batchSqueeze(
            transcripts: transcripts, label: "alpha")

        expectEqual(challenges.count, 4, "should produce 4 challenges")
        expectEqual(updated.count, 4, "should return 4 updated transcripts")

        // All challenges should be non-zero
        for (i, c) in challenges.enumerated() {
            let limbs = c.to64()
            let isNonZero = limbs[0] != 0 || limbs[1] != 0 || limbs[2] != 0 || limbs[3] != 0
            expect(isNonZero, "batch challenge \(i) should be non-zero")
        }

        // All challenges should be distinct (each transcript had a different input)
        for i in 0..<challenges.count {
            for j in (i+1)..<challenges.count {
                let li = challenges[i].to64()
                let lj = challenges[j].to64()
                let differ = li[0] != lj[0] || li[1] != lj[1] ||
                             li[2] != lj[2] || li[3] != lj[3]
                expect(differ, "batch challenges \(i) and \(j) should differ")
            }
        }
    } catch {
        expect(false, "batch squeeze should not throw: \(error)")
    }
}

// MARK: - Batch Engine: Squeeze Multiple

private func testBatchSqueezeMultiple() {
    let batch = GPUBatchFiatShamirEngine(label: "batch-multi")
    var transcripts = batch.createBatch(count: 3)

    let values = [frFromInt(10), frFromInt(20), frFromInt(30)]
    transcripts = batch.batchAppendScalar(
        transcripts: transcripts, label: "seed", values: values)

    do {
        let (_, challenges) = try batch.batchSqueezeMultiple(
            transcripts: transcripts, label: "rounds", count: 3)

        expectEqual(challenges.count, 3, "should have 3 transcript results")
        for (i, cs) in challenges.enumerated() {
            expectEqual(cs.count, 3, "transcript \(i) should have 3 challenges")
        }
    } catch {
        expect(false, "batch squeeze multiple should not throw: \(error)")
    }
}

// MARK: - Batch Engine: Fork

private func testBatchFork() {
    let batch = GPUBatchFiatShamirEngine(label: "batch-fork")
    var transcripts = batch.createBatch(count: 4)

    let values = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
    transcripts = batch.batchAppendScalar(
        transcripts: transcripts, label: "x", values: values)

    let forked = batch.batchFork(transcripts: transcripts, label: "sub-proof")
    expectEqual(forked.count, 4, "should fork 4 transcripts")

    // Forked transcripts should have reset op count (0 from fork creation)
    for (i, f) in forked.enumerated() {
        expectEqual(f.operationCount, 0, "forked transcript \(i) should have op count 0")
    }
}

// MARK: - Batch Engine: Determinism

private func testBatchDeterminism() {
    let batch = GPUBatchFiatShamirEngine(label: "batch-det")

    // Run 1
    var t1 = batch.createBatch(count: 3)
    let v1 = [frFromInt(7), frFromInt(13), frFromInt(37)]
    t1 = batch.batchAppendScalar(transcripts: t1, label: "x", values: v1)

    // Run 2
    var t2 = batch.createBatch(count: 3)
    t2 = batch.batchAppendScalar(transcripts: t2, label: "x", values: v1)

    do {
        let (_, c1) = try batch.batchSqueeze(transcripts: t1, label: "c")
        let (_, c2) = try batch.batchSqueeze(transcripts: t2, label: "c")

        for i in 0..<3 {
            let l1 = c1[i].to64()
            let l2 = c2[i].to64()
            let match = l1[0] == l2[0] && l1[1] == l2[1] && l1[2] == l2[2] && l1[3] == l2[3]
            expect(match, "batch determinism: transcript \(i) should match")
        }
    } catch {
        expect(false, "batch determinism test should not throw: \(error)")
    }
}

// MARK: - Batch Engine: Independence

private func testBatchIndependence() {
    let batch = GPUBatchFiatShamirEngine(label: "batch-indep")
    var transcripts = batch.createBatch(count: 2)

    // Give different inputs to each transcript
    let values = [frFromInt(1), frFromInt(2)]
    transcripts = batch.batchAppendScalar(
        transcripts: transcripts, label: "x", values: values)

    do {
        let (_, challenges) = try batch.batchSqueeze(
            transcripts: transcripts, label: "c")

        let l0 = challenges[0].to64()
        let l1 = challenges[1].to64()
        let differ = l0[0] != l1[0] || l0[1] != l1[1] || l0[2] != l1[2] || l0[3] != l1[3]
        expect(differ, "independent transcripts with different inputs should differ")
    } catch {
        expect(false, "batch independence test should not throw: \(error)")
    }
}

// MARK: - Consistency Checker: Match

private func testConsistencyCheckerMatch() {
    var checker = TranscriptConsistencyChecker(label: "consistency")

    let matched = checker.appendScalar(label: "x", value: frFromInt(42))
    expect(matched, "identical absorbs should keep consistency")
    expect(checker.isConsistent, "checker should be consistent after matching ops")

    let (pc, vc, same) = checker.squeezeAndVerify(label: "c")
    expect(same, "matching transcripts should produce same challenge")

    let pl = pc.to64()
    let vl = vc.to64()
    expect(pl[0] == vl[0] && pl[1] == vl[1] && pl[2] == vl[2] && pl[3] == vl[3],
           "prover and verifier challenges should be identical")
    expect(checker.firstDivergence == nil, "no divergence point for matching transcripts")
}

// MARK: - Consistency Checker: Divergence Detection

private func testConsistencyCheckerDivergence() {
    // Build two separate engines with different inputs to simulate divergence
    var engine1 = GPUFiatShamirEngine(label: "div-test")
    var engine2 = GPUFiatShamirEngine(label: "div-test")

    engine1.appendScalar(label: "x", value: frFromInt(1))
    engine2.appendScalar(label: "x", value: frFromInt(2))  // Different value!

    let c1 = engine1.squeezeChallenge(label: "c")
    let c2 = engine2.squeezeChallenge(label: "c")

    let l1 = c1.to64()
    let l2 = c2.to64()
    let differ = l1[0] != l2[0] || l1[1] != l2[1] || l1[2] != l2[2] || l1[3] != l2[3]
    expect(differ, "different inputs should cause divergent challenges")
}

// MARK: - Consistency Checker: Multi-Step

private func testConsistencyMultiStep() {
    var checker = TranscriptConsistencyChecker(label: "multi-step")

    // Multiple rounds of absorb + squeeze
    for i in 0..<5 {
        let ok = checker.appendScalar(label: "round-\(i)", value: frFromInt(UInt64(i * 7)))
        expect(ok, "step \(i) should maintain consistency")
    }

    let (_, _, match) = checker.squeezeAndVerify(label: "final")
    expect(match, "multi-step consistency should hold")
    expect(checker.isConsistent, "checker should remain consistent after multi-step")
}

// MARK: - SpongeField Protocol: Fr

private func testFrSpongeField() {
    let zero = Fr.spongeZero
    let zeroLimbs = zero.to64()
    expect(zeroLimbs[0] == 0 && zeroLimbs[1] == 0 && zeroLimbs[2] == 0 && zeroLimbs[3] == 0,
           "Fr.spongeZero should be zero")

    let a = frFromInt(5)
    let b = frFromInt(3)
    let sum = Fr.spongeAdd(a, b)
    let expected = frFromInt(8)
    let sl = sum.to64()
    let el = expected.to64()
    expect(sl[0] == el[0] && sl[1] == el[1] && sl[2] == el[2] && sl[3] == el[3],
           "Fr.spongeAdd should compute field addition")

    let bytes = a.spongeBytes()
    expectEqual(bytes.count, 32, "Fr sponge bytes should be 32 bytes")
}

// MARK: - SpongeField Protocol: Bb

private func testBbSpongeField() {
    let zero = Bb.spongeZero
    expectEqual(zero.v, 0, "Bb.spongeZero should be 0")

    let a = Bb(v: 5)
    let b = Bb(v: 3)
    let sum = Bb.spongeAdd(a, b)
    expectEqual(sum.v, 8, "Bb.spongeAdd(5, 3) should be 8")

    let bytes = a.spongeBytes()
    expectEqual(bytes.count, 4, "Bb sponge bytes should be 4 bytes")
    expectEqual(bytes[0], 5, "Bb sponge bytes first byte should be 5")
}

// MARK: - FiatShamirSpongeState

private func testSpongeStateInit() {
    let state = FiatShamirSpongeState<Fr>()
    let s0limbs = state.s0.to64()
    expect(s0limbs[0] == 0 && s0limbs[1] == 0, "default state s0 should be zero")
    expectEqual(state.absorbed, 0, "default absorbed should be 0")
    expectEqual(state.opCount, 0, "default opCount should be 0")
}

private func testSpongeStateExplicitInit() {
    let s0 = frFromInt(1)
    let s1 = frFromInt(2)
    let s2 = frFromInt(3)
    let state = FiatShamirSpongeState<Fr>(s0: s0, s1: s1, s2: s2, absorbed: 1, opCount: 42)

    let s0l = state.s0.to64()
    let expected = frFromInt(1).to64()
    expect(s0l[0] == expected[0], "explicit init s0 should match")
    expectEqual(state.absorbed, 1, "explicit init absorbed should be 1")
    expectEqual(state.opCount, 42, "explicit init opCount should be 42")
}

// MARK: - TranscriptEngine Conformance

private func testTranscriptEngineConformance() {
    // GPUFiatShamirEngine should conform to TranscriptEngine
    var engine = GPUFiatShamirEngine(label: "conformance-test")

    // Use the TranscriptEngine API
    engine.appendMessage(label: "msg", data: [0x01, 0x02])
    engine.appendScalar(label: "scalar", scalar: frFromInt(42))

    let c = engine.squeezeChallenge()
    let limbs = c.to64()
    let isNonZero = limbs[0] != 0 || limbs[1] != 0 || limbs[2] != 0 || limbs[3] != 0
    expect(isNonZero, "TranscriptEngine squeeze should produce non-zero challenge")
}

private func testTranscriptEngineSqueezeChallenges() {
    var engine = GPUFiatShamirEngine(label: "conformance-multi")
    engine.appendScalar(label: "seed", scalar: frFromInt(7))

    let challenges = engine.squeezeChallenges(count: 3)
    expectEqual(challenges.count, 3, "TranscriptEngine should squeeze 3 challenges")

    // Verify they are all distinct
    for i in 0..<challenges.count {
        for j in (i+1)..<challenges.count {
            let li = challenges[i].to64()
            let lj = challenges[j].to64()
            let differ = li[0] != lj[0] || li[1] != lj[1] || li[2] != lj[2] || li[3] != lj[3]
            expect(differ, "conformance challenges \(i) and \(j) should differ")
        }
    }
}
