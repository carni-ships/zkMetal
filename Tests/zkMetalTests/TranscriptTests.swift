import zkMetal

func runTranscriptTests() {
    // =========================================================================
    // TranscriptEngine protocol tests (shared across all backends)
    // =========================================================================

    suite("TranscriptEngine — KeccakTranscript")
    testTranscriptEngineTyped { (label: String) -> KeccakTranscript in KeccakTranscript(label: label) }

    suite("TranscriptEngine — Poseidon2Transcript")
    testTranscriptEngineTyped { (label: String) -> Poseidon2Transcript in Poseidon2Transcript(label: label) }

    suite("TranscriptEngine — Blake3Transcript")
    testTranscriptEngineTyped { (label: String) -> Blake3Transcript in Blake3Transcript(label: label) }

    suite("TranscriptEngine — MerlinTranscript")
    testMerlinTranscript()

    suite("TranscriptEngine — Transcript (class)")
    testTranscriptClass()

    suite("TranscriptEngine — Replay Protection")
    testReplayProtection()

    suite("TranscriptEngine — Fork Isolation")
    testForkIsolation()

    suite("TranscriptEngine — Domain Separation")
    testDomainSeparation()

    suite("TranscriptEngine — Blake3 Direct (no existential)")
    testBlake3Direct()

    suite("TranscriptEngine — Cross-Backend Independence")
    testCrossBackendIndependence()
}

// MARK: - Generic TranscriptEngine tests

/// Test core protocol behavior using concrete generic types (avoids existential mutation issues).
private func testTranscriptEngineTyped<T: TranscriptEngine>(_ makeTranscript: (String) -> T) {
    // Determinism: same inputs -> same output
    do {
        var t1 = makeTranscript("test-determinism")
        var t2 = makeTranscript("test-determinism")
        let data: [UInt8] = [1, 2, 3, 4, 5]
        t1.appendMessage(label: "msg", data: data)
        t2.appendMessage(label: "msg", data: data)
        let c1 = t1.squeezeChallenge()
        let c2 = t2.squeezeChallenge()
        expect(frEq(c1, c2), "Deterministic: same inputs -> same challenge")
    }

    // Non-trivial output: challenge is not zero
    do {
        var t = makeTranscript("test-nontrivial")
        t.appendMessage(label: "data", data: [42])
        let c = t.squeezeChallenge()
        let limbs = frToInt(c)
        let nonZero = limbs[0] != 0 || limbs[1] != 0 || limbs[2] != 0 || limbs[3] != 0
        expect(nonZero, "Non-trivial: challenge is not zero")
    }

    // Different data -> different challenges
    do {
        var t1 = makeTranscript("test-diff")
        var t2 = makeTranscript("test-diff")
        t1.appendMessage(label: "msg", data: [1, 2, 3])
        t2.appendMessage(label: "msg", data: [4, 5, 6])
        let c1 = t1.squeezeChallenge()
        let c2 = t2.squeezeChallenge()
        expect(!frEq(c1, c2), "Different data -> different challenges")
    }

    // Different labels -> different challenges
    do {
        var t1 = makeTranscript("test-labels")
        var t2 = makeTranscript("test-labels")
        t1.appendMessage(label: "alpha", data: [1, 2, 3])
        t2.appendMessage(label: "beta", data: [1, 2, 3])
        let c1 = t1.squeezeChallenge()
        let c2 = t2.squeezeChallenge()
        expect(!frEq(c1, c2), "Different labels -> different challenges")
    }

    // Multiple squeezes produce distinct challenges
    do {
        var t = makeTranscript("test-multi-squeeze")
        t.appendMessage(label: "data", data: [1, 2, 3])
        var challenges = [Fr]()
        for _ in 0..<5 {
            challenges.append(t.squeezeChallenge())
        }
        var allDistinct = true
        for i in 0..<5 {
            for j in (i + 1)..<5 {
                if frEq(challenges[i], challenges[j]) {
                    allDistinct = false
                }
            }
        }
        expect(allDistinct, "Multiple squeezes produce distinct challenges")
    }

    // appendScalar round-trip
    do {
        var t1 = makeTranscript("test-scalar")
        var t2 = makeTranscript("test-scalar")
        let s = frFromInt(12345)
        t1.appendScalar(label: "s", scalar: s)
        t2.appendScalar(label: "s", scalar: s)
        let c1 = t1.squeezeChallenge()
        let c2 = t2.squeezeChallenge()
        expect(frEq(c1, c2), "appendScalar deterministic")
    }

    // appendPoint
    do {
        var t1 = makeTranscript("test-point")
        var t2 = makeTranscript("test-point")
        let p = pointIdentity()
        t1.appendPoint(label: "P", point: p)
        t2.appendPoint(label: "P", point: p)
        let c1 = t1.squeezeChallenge()
        let c2 = t2.squeezeChallenge()
        expect(frEq(c1, c2), "appendPoint deterministic")
    }

    // Fork produces independent challenges (uses existential return, so squeeze individually)
    do {
        var t = makeTranscript("test-fork")
        t.appendMessage(label: "data", data: [1, 2, 3])
        // Fork returns any TranscriptEngine, use squeezeChallenge through existential
        // (single call is fine for existential mutation)
        var child1 = t.fork(label: "child-A")
        var child2 = t.fork(label: "child-B")
        let c1 = child1.squeezeChallenge()
        let c2 = child2.squeezeChallenge()
        expect(!frEq(c1, c2), "Fork with different labels -> different challenges")
    }

    // Fork preserves parent state
    do {
        var t = makeTranscript("test-fork-preserve")
        t.appendMessage(label: "shared", data: [1, 2, 3])
        _ = t.fork(label: "child")
        // Parent can still squeeze normally
        let c = t.squeezeChallenge()
        let limbs = frToInt(c)
        let nonZero = limbs[0] != 0 || limbs[1] != 0 || limbs[2] != 0 || limbs[3] != 0
        expect(nonZero, "Fork does not corrupt parent state")
    }

    // Operation count increases
    do {
        var t = makeTranscript("test-opcount")
        let initial = t.operationCount
        t.appendMessage(label: "a", data: [1])
        let afterAppend = t.operationCount
        _ = t.squeezeChallenge()
        let afterSqueeze = t.operationCount
        expect(afterAppend > initial, "Operation count increases after append")
        expect(afterSqueeze > afterAppend, "Operation count increases after squeeze")
    }
}

// MARK: - Merlin-specific tests

private func testMerlinTranscript() {
    // Basic determinism
    do {
        var t1 = MerlinTranscript(label: "test")
        var t2 = MerlinTranscript(label: "test")
        t1.appendMessage(label: "msg", data: [1, 2, 3])
        t2.appendMessage(label: "msg", data: [1, 2, 3])
        let c1 = t1.squeezeChallenge()
        let c2 = t2.squeezeChallenge()
        expect(frEq(c1, c2), "Merlin deterministic")
    }

    // Different protocol labels -> different output
    do {
        var t1 = MerlinTranscript(label: "protocol-A")
        var t2 = MerlinTranscript(label: "protocol-B")
        t1.appendMessage(label: "data", data: [1])
        t2.appendMessage(label: "data", data: [1])
        let c1 = t1.squeezeChallenge()
        let c2 = t2.squeezeChallenge()
        expect(!frEq(c1, c2), "Merlin different protocol labels -> different challenges")
    }

    // challengeBytes API
    do {
        var t1 = MerlinTranscript(label: "test-bytes")
        var t2 = MerlinTranscript(label: "test-bytes")
        t1.appendMessage(label: "data", data: [42])
        t2.appendMessage(label: "data", data: [42])
        let b1 = t1.challengeBytes(label: "c", count: 32)
        let b2 = t2.challengeBytes(label: "c", count: 32)
        expect(b1 == b2, "Merlin challengeBytes deterministic")
        expect(b1.count == 32, "Merlin challengeBytes correct length")
        expect(b1 != [UInt8](repeating: 0, count: 32), "Merlin challengeBytes non-zero")
    }

    // Merlin append order matters
    do {
        var t1 = MerlinTranscript(label: "order-test")
        var t2 = MerlinTranscript(label: "order-test")
        t1.appendMessage(label: "A", data: [1])
        t1.appendMessage(label: "B", data: [2])
        t2.appendMessage(label: "B", data: [2])
        t2.appendMessage(label: "A", data: [1])
        let c1 = t1.squeezeChallenge()
        let c2 = t2.squeezeChallenge()
        expect(!frEq(c1, c2), "Merlin: different append order -> different challenges")
    }

    // Merlin operation count
    do {
        var t = MerlinTranscript(label: "opcount")
        let initial = t.operationCount
        t.appendMessage(label: "x", data: [1])
        expect(t.operationCount > initial, "Merlin operation count increments on append")
        _ = t.challengeBytes(label: "c", count: 32)
        expect(t.operationCount > initial + 1, "Merlin operation count increments on squeeze")
    }

    // Merlin appendScalar
    do {
        var t1 = MerlinTranscript(label: "scalar-test")
        var t2 = MerlinTranscript(label: "scalar-test")
        let s = frFromInt(9999)
        t1.appendScalar(label: "s", scalar: s)
        t2.appendScalar(label: "s", scalar: s)
        let c1 = t1.squeezeChallenge()
        let c2 = t2.squeezeChallenge()
        expect(frEq(c1, c2), "Merlin appendScalar deterministic")
    }

    // Merlin fork
    do {
        var t = MerlinTranscript(label: "fork-test")
        t.appendMessage(label: "shared", data: [1, 2, 3])
        var f1 = t.fork(label: "branch-A")
        var f2 = t.fork(label: "branch-B")
        let c1 = f1.squeezeChallenge()
        let c2 = f2.squeezeChallenge()
        expect(!frEq(c1, c2), "Merlin fork produces independent challenges")
    }
}

// MARK: - Transcript class tests (backward compatibility)

private func testTranscriptClass() {
    // Original absorb/squeeze API still works
    do {
        let t = Transcript(label: "legacy", backend: .poseidon2)
        t.absorb(frFromInt(42))
        let c = t.squeeze()
        let limbs = frToInt(c)
        let nonZero = limbs[0] != 0 || limbs[1] != 0 || limbs[2] != 0 || limbs[3] != 0
        expect(nonZero, "Legacy API: absorb/squeeze works")
    }

    // Protocol conformance: appendMessage + squeezeChallenge
    do {
        let t = Transcript(label: "protocol-test", backend: .keccak256)
        t.appendMessage(label: "data", data: [1, 2, 3])
        let c = t.squeezeChallenge()
        let limbs = frToInt(c)
        let nonZero = limbs[0] != 0 || limbs[1] != 0 || limbs[2] != 0 || limbs[3] != 0
        expect(nonZero, "Protocol conformance: appendMessage + squeezeChallenge")
    }

    // squeezeN still works
    do {
        let t = Transcript(label: "multi", backend: .poseidon2)
        t.absorb(frFromInt(1))
        let challenges = t.squeezeN(3)
        expect(challenges.count == 3, "squeezeN returns correct count")
        expect(!frEq(challenges[0], challenges[1]), "squeezeN produces distinct values")
    }

    // Operation count on class
    do {
        let t = Transcript(label: "opcount", backend: .keccak256)
        let c0 = t.operationCount
        t.absorb(frFromInt(1))
        let c1 = t.operationCount
        _ = t.squeeze()
        let c2 = t.operationCount
        expect(c1 > c0, "Class operation count increases on absorb")
        expect(c2 > c1, "Class operation count increases on squeeze")
    }
}

// MARK: - Replay protection tests

private func testReplayProtection() {
    // Operation counter monotonically increases
    do {
        var t: KeccakTranscript = KeccakTranscript(label: "replay-test")
        var counts = [UInt64]()
        counts.append(t.operationCount)
        t.appendMessage(label: "a", data: [1])
        counts.append(t.operationCount)
        t.appendScalar(label: "s", scalar: frFromInt(42))
        counts.append(t.operationCount)
        _ = t.squeezeChallenge()
        counts.append(t.operationCount)
        _ = t.squeezeChallenges(count: 3)
        counts.append(t.operationCount)

        var monotonic = true
        for i in 1..<counts.count {
            if counts[i] <= counts[i - 1] {
                monotonic = false
            }
        }
        expect(monotonic, "Operation counter is monotonically increasing")
    }

    // Counter affects challenge output (replay protection)
    // Two transcripts with same data but squeezing at different points
    // should produce different results for same-indexed challenges
    do {
        var t1: KeccakTranscript = KeccakTranscript(label: "replay-diff")
        var t2: KeccakTranscript = KeccakTranscript(label: "replay-diff")

        t1.appendMessage(label: "data", data: [1, 2, 3])
        t2.appendMessage(label: "data", data: [1, 2, 3])

        // t1 squeezes immediately
        let c1 = t1.squeezeChallenge()

        // t2 does an extra append before squeezing
        t2.appendMessage(label: "extra", data: [4, 5, 6])
        let c2 = t2.squeezeChallenge()

        expect(!frEq(c1, c2), "Extra operations change subsequent challenges")
    }
}

// MARK: - Fork isolation tests

private func testForkIsolation() {
    // Child operations don't affect parent
    do {
        var t: Poseidon2Transcript = Poseidon2Transcript(label: "fork-isolation")
        t.appendMessage(label: "shared", data: [1, 2, 3])

        // Snapshot parent state by creating two identical forks
        var ref = t.fork(label: "ref") as! Poseidon2Transcript
        var child = t.fork(label: "child") as! Poseidon2Transcript

        // Modify child
        child.appendMessage(label: "child-data", data: [99])
        _ = child.squeezeChallenge()

        // Reference should still match a fresh fork with same label
        var ref2 = t.fork(label: "ref") as! Poseidon2Transcript
        let c1 = ref.squeezeChallenge()
        let c2 = ref2.squeezeChallenge()
        expect(frEq(c1, c2), "Fork isolation: child modifications don't affect sibling forks")
    }

    // Same fork label -> same challenges
    do {
        var t: Blake3Transcript = Blake3Transcript(label: "fork-same")
        t.appendMessage(label: "data", data: [7, 8, 9])
        var f1 = t.fork(label: "same-label") as! Blake3Transcript
        var f2 = t.fork(label: "same-label") as! Blake3Transcript
        let c1 = f1.squeezeChallenge()
        let c2 = f2.squeezeChallenge()
        expect(frEq(c1, c2), "Same fork label -> same challenges")
    }

    // Different fork labels -> different challenges
    do {
        var t: Blake3Transcript = Blake3Transcript(label: "fork-diff")
        t.appendMessage(label: "data", data: [7, 8, 9])
        var f1 = t.fork(label: "label-A")
        var f2 = t.fork(label: "label-B")
        let c1 = f1.squeezeChallenge()
        let c2 = f2.squeezeChallenge()
        expect(!frEq(c1, c2), "Different fork labels -> different challenges")
    }
}

// MARK: - Domain separation tests

private func testDomainSeparation() {
    // Different protocol labels produce different transcripts
    do {
        var t1: KeccakTranscript = KeccakTranscript(label: "Plonk-v1")
        var t2: KeccakTranscript = KeccakTranscript(label: "Groth16")
        // Same data
        t1.appendMessage(label: "data", data: [1, 2, 3])
        t2.appendMessage(label: "data", data: [1, 2, 3])
        let c1 = t1.squeezeChallenge()
        let c2 = t2.squeezeChallenge()
        expect(!frEq(c1, c2), "Different protocol labels -> different challenges")
    }

    // Empty vs non-empty data
    do {
        var t1: Poseidon2Transcript = Poseidon2Transcript(label: "empty-test")
        var t2: Poseidon2Transcript = Poseidon2Transcript(label: "empty-test")
        t1.appendMessage(label: "msg", data: [])
        t2.appendMessage(label: "msg", data: [0])
        let c1 = t1.squeezeChallenge()
        let c2 = t2.squeezeChallenge()
        expect(!frEq(c1, c2), "Empty vs non-empty data -> different challenges")
    }

    // Label collision prevention: "ab" + "c" vs "a" + "bc"
    do {
        var t1: KeccakTranscript = KeccakTranscript(label: "collision-test")
        var t2: KeccakTranscript = KeccakTranscript(label: "collision-test")
        t1.appendMessage(label: "ab", data: Array("c".utf8))
        t2.appendMessage(label: "a", data: Array("bc".utf8))
        let c1 = t1.squeezeChallenge()
        let c2 = t2.squeezeChallenge()
        expect(!frEq(c1, c2), "Length-prefixed labels prevent collisions")
    }
}

// MARK: - Blake3 direct tests (no existential)

private func testBlake3Direct() {
    var t: Blake3Transcript = Blake3Transcript(label: "test-multi-squeeze")
    t.appendMessage(label: "data", data: [1, 2, 3])
    // Call squeezeChallenge directly 5 times
    var challenges = [Fr]()
    for _ in 0..<5 {
        challenges.append(t.squeezeChallenge())
    }
    var allDistinct = true
    for i in 0..<5 {
        for j in (i + 1)..<5 {
            if frEq(challenges[i], challenges[j]) {
                allDistinct = false
            }
        }
    }
    expect(allDistinct, "Blake3 direct: 5 squeezes produce distinct challenges")

    // Also test squeezeChallenges directly on typed value
    var t2: Blake3Transcript = Blake3Transcript(label: "test-multi-squeeze")
    t2.appendMessage(label: "data", data: [1, 2, 3])
    let challenges2 = t2.squeezeChallenges(count: 5)
    var allDistinct2 = true
    for i in 0..<5 {
        for j in (i + 1)..<5 {
            if frEq(challenges2[i], challenges2[j]) {
                allDistinct2 = false
            }
        }
    }
    expect(allDistinct2, "Blake3 typed squeezeChallenges: 5 distinct challenges")
}

// MARK: - Cross-backend independence tests

private func testCrossBackendIndependence() {
    // Different backends produce different outputs (they are independent constructions)
    do {
        var keccak: KeccakTranscript = KeccakTranscript(label: "cross-test")
        var poseidon: Poseidon2Transcript = Poseidon2Transcript(label: "cross-test")
        var blake3: Blake3Transcript = Blake3Transcript(label: "cross-test")

        keccak.appendMessage(label: "data", data: [1, 2, 3])
        poseidon.appendMessage(label: "data", data: [1, 2, 3])
        blake3.appendMessage(label: "data", data: [1, 2, 3])

        let ck = keccak.squeezeChallenge()
        let cp = poseidon.squeezeChallenge()
        let cb = blake3.squeezeChallenge()

        // All three should differ (different hash constructions)
        expect(!frEq(ck, cp), "Keccak != Poseidon2")
        expect(!frEq(ck, cb), "Keccak != Blake3")
        expect(!frEq(cp, cb), "Poseidon2 != Blake3")
    }

    // Merlin differs from all others too
    do {
        var merlin = MerlinTranscript(label: "cross-test")
        var keccak: KeccakTranscript = KeccakTranscript(label: "cross-test")

        merlin.appendMessage(label: "data", data: [1, 2, 3])
        keccak.appendMessage(label: "data", data: [1, 2, 3])

        let cm = merlin.squeezeChallenge()
        let ck = keccak.squeezeChallenge()

        expect(!frEq(cm, ck), "Merlin != Keccak (different constructions)")
    }

    // Large data absorption works correctly
    do {
        var t: KeccakTranscript = KeccakTranscript(label: "large-data")
        let bigData = [UInt8](repeating: 0xAB, count: 10000)
        t.appendMessage(label: "big", data: bigData)
        let c = t.squeezeChallenge()
        let limbs = frToInt(c)
        let nonZero = limbs[0] != 0 || limbs[1] != 0 || limbs[2] != 0 || limbs[3] != 0
        expect(nonZero, "Large data absorption produces non-zero challenge")
    }

    // Squeeze many challenges (stress test)
    do {
        var t: Poseidon2Transcript = Poseidon2Transcript(label: "stress")
        t.appendMessage(label: "seed", data: [42])
        let challenges = t.squeezeChallenges(count: 100)
        expect(challenges.count == 100, "Squeeze 100 challenges")

        // Check uniqueness of first 10
        var allUnique = true
        for i in 0..<10 {
            for j in (i + 1)..<10 {
                if frEq(challenges[i], challenges[j]) {
                    allUnique = false
                }
            }
        }
        expect(allUnique, "100 challenges: first 10 are all unique")
    }
}
