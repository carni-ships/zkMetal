import zkMetal

/// Hardened Fiat-Shamir transcript tests covering domain separation,
/// absorb ordering, length prefix security, fork independence,
/// cross-hash consistency, and replay determinism.
public func runTranscriptHardeningTests() {
    suite("TranscriptHardening — Domain Separation")
    testDomainSeparationLabels()

    suite("TranscriptHardening — Absorb Order")
    testAbsorbOrderMatters()

    suite("TranscriptHardening — Length Prefix Security")
    testLengthPrefixSecurity()

    suite("TranscriptHardening — Fork Independence")
    testForkIndependence()

    suite("TranscriptHardening — Cross-Hash Consistency")
    testCrossHashConsistency()

    suite("TranscriptHardening — Replay Determinism")
    testReplayDeterminism()

    suite("TranscriptHardening — TranscriptProtocol Factory")
    testTranscriptProtocolFactory()

    suite("TranscriptHardening — Protocol Begin")
    testProtocolBegin()
}

// MARK: - Domain Separation: same data with different labels -> different challenges

private func testDomainSeparationLabels() {
    // absorbScalar with different labels
    do {
        var t1 = DomainSeparatedTranscript(hash: .keccak)
        var t2 = DomainSeparatedTranscript(hash: .keccak)
        let s = frFromInt(42)
        t1.absorbScalar("alpha", s)
        t2.absorbScalar("beta", s)
        let c1 = t1.squeezeChallenge("challenge")
        let c2 = t2.squeezeChallenge("challenge")
        expect(!frEq(c1, c2), "Different scalar labels -> different challenges")
    }

    // absorbBytes with different labels
    do {
        var t1 = DomainSeparatedTranscript(hash: .poseidon2)
        var t2 = DomainSeparatedTranscript(hash: .poseidon2)
        let data: [UInt8] = [1, 2, 3, 4, 5]
        t1.absorbBytes("input_a", data)
        t2.absorbBytes("input_b", data)
        let c1 = t1.squeezeChallenge("out")
        let c2 = t2.squeezeChallenge("out")
        expect(!frEq(c1, c2), "Different bytes labels -> different challenges")
    }

    // absorbPoint with different labels
    do {
        var t1 = DomainSeparatedTranscript(hash: .blake3)
        var t2 = DomainSeparatedTranscript(hash: .blake3)
        let p = pointIdentity()
        t1.absorbPoint("commit_a", p)
        t2.absorbPoint("commit_b", p)
        let c1 = t1.squeezeChallenge("ch")
        let c2 = t2.squeezeChallenge("ch")
        expect(!frEq(c1, c2), "Different point labels -> different challenges")
    }

    // squeezeChallenge with different labels from same state
    do {
        var t1 = DomainSeparatedTranscript(hash: .keccak)
        var t2 = DomainSeparatedTranscript(hash: .keccak)
        let data: [UInt8] = [10, 20, 30]
        t1.absorbBytes("data", data)
        t2.absorbBytes("data", data)
        let c1 = t1.squeezeChallenge("alpha")
        let c2 = t2.squeezeChallenge("beta")
        expect(!frEq(c1, c2), "Different squeeze labels -> different challenges")
    }
}

// MARK: - Absorb order matters: absorb(a, b) != absorb(b, a)

private func testAbsorbOrderMatters() {
    // Two scalars in different order
    do {
        var t1 = DomainSeparatedTranscript(hash: .keccak)
        var t2 = DomainSeparatedTranscript(hash: .keccak)
        let a = frFromInt(111)
        let b = frFromInt(222)
        t1.absorbScalar("x", a)
        t1.absorbScalar("y", b)
        t2.absorbScalar("x", b)
        t2.absorbScalar("y", a)
        let c1 = t1.squeezeChallenge("out")
        let c2 = t2.squeezeChallenge("out")
        expect(!frEq(c1, c2), "Scalar absorb order matters (different values)")
    }

    // Two byte arrays in different order (same labels)
    do {
        var t1 = DomainSeparatedTranscript(hash: .poseidon2)
        var t2 = DomainSeparatedTranscript(hash: .poseidon2)
        t1.absorbBytes("first", [1, 2, 3])
        t1.absorbBytes("second", [4, 5, 6])
        t2.absorbBytes("first", [4, 5, 6])
        t2.absorbBytes("second", [1, 2, 3])
        let c1 = t1.squeezeChallenge("out")
        let c2 = t2.squeezeChallenge("out")
        expect(!frEq(c1, c2), "Bytes absorb order matters (swapped payloads)")
    }

    // Same label but different absorb order of operations
    do {
        var t1 = DomainSeparatedTranscript(hash: .blake3)
        var t2 = DomainSeparatedTranscript(hash: .blake3)
        t1.absorbBytes("A", [1])
        t1.absorbBytes("B", [2])
        t2.absorbBytes("B", [2])
        t2.absorbBytes("A", [1])
        let c1 = t1.squeezeChallenge("out")
        let c2 = t2.squeezeChallenge("out")
        expect(!frEq(c1, c2), "Operation order matters (A then B != B then A)")
    }
}

// MARK: - Length prefix: absorb([1,2]) + absorb([3]) != absorb([1]) + absorb([2,3])

private func testLengthPrefixSecurity() {
    // Keccak backend
    do {
        var t1 = DomainSeparatedTranscript(hash: .keccak)
        var t2 = DomainSeparatedTranscript(hash: .keccak)
        t1.absorbBytes("a", [1, 2])
        t1.absorbBytes("b", [3])
        t2.absorbBytes("a", [1])
        t2.absorbBytes("b", [2, 3])
        let c1 = t1.squeezeChallenge("out")
        let c2 = t2.squeezeChallenge("out")
        expect(!frEq(c1, c2), "Keccak: length prefix prevents [1,2]+[3] == [1]+[2,3]")
    }

    // Poseidon2 backend
    do {
        var t1 = DomainSeparatedTranscript(hash: .poseidon2)
        var t2 = DomainSeparatedTranscript(hash: .poseidon2)
        t1.absorbBytes("a", [1, 2])
        t1.absorbBytes("b", [3])
        t2.absorbBytes("a", [1])
        t2.absorbBytes("b", [2, 3])
        let c1 = t1.squeezeChallenge("out")
        let c2 = t2.squeezeChallenge("out")
        expect(!frEq(c1, c2), "Poseidon2: length prefix prevents [1,2]+[3] == [1]+[2,3]")
    }

    // Blake3 backend
    do {
        var t1 = DomainSeparatedTranscript(hash: .blake3)
        var t2 = DomainSeparatedTranscript(hash: .blake3)
        t1.absorbBytes("a", [1, 2])
        t1.absorbBytes("b", [3])
        t2.absorbBytes("a", [1])
        t2.absorbBytes("b", [2, 3])
        let c1 = t1.squeezeChallenge("out")
        let c2 = t2.squeezeChallenge("out")
        expect(!frEq(c1, c2), "Blake3: length prefix prevents [1,2]+[3] == [1]+[2,3]")
    }

    // Empty vs single-byte: absorb([]) + absorb([0]) != absorb([0]) + absorb([])
    do {
        var t1 = DomainSeparatedTranscript(hash: .keccak)
        var t2 = DomainSeparatedTranscript(hash: .keccak)
        t1.absorbBytes("a", [])
        t1.absorbBytes("b", [0])
        t2.absorbBytes("a", [0])
        t2.absorbBytes("b", [])
        let c1 = t1.squeezeChallenge("out")
        let c2 = t2.squeezeChallenge("out")
        expect(!frEq(c1, c2), "Length prefix: empty+[0] != [0]+empty")
    }
}

// MARK: - Fork independence: forked transcript diverges from parent

private func testForkIndependence() {
    // Fork diverges from parent
    do {
        var parent = DomainSeparatedTranscript(hash: .keccak)
        parent.absorbBytes("shared", [1, 2, 3])
        var child = parent.fork("child-0")
        let parentChallenge = parent.squeezeChallenge("out")
        let childChallenge = child.squeezeChallenge("out")
        expect(!frEq(parentChallenge, childChallenge), "Fork diverges from parent")
    }

    // Two forks with different labels produce different challenges
    do {
        var parent = DomainSeparatedTranscript(hash: .poseidon2)
        parent.absorbBytes("shared", [7, 8, 9])
        var fork1 = parent.fork("branch-A")
        var fork2 = parent.fork("branch-B")
        let c1 = fork1.squeezeChallenge("out")
        let c2 = fork2.squeezeChallenge("out")
        expect(!frEq(c1, c2), "Different fork labels -> different challenges")
    }

    // Two forks with same label produce same challenges (determinism)
    do {
        var parent = DomainSeparatedTranscript(hash: .blake3)
        parent.absorbBytes("shared", [7, 8, 9])
        var fork1 = parent.fork("same-label")
        var fork2 = parent.fork("same-label")
        let c1 = fork1.squeezeChallenge("out")
        let c2 = fork2.squeezeChallenge("out")
        expect(frEq(c1, c2), "Same fork label -> same challenges")
    }

    // Child modification does not affect sibling fork
    do {
        var parent = DomainSeparatedTranscript(hash: .keccak)
        parent.absorbBytes("shared", [1, 2, 3])
        var child1 = parent.fork("child")
        var child2 = parent.fork("child")
        // Mutate child1
        child1.absorbBytes("extra", [99, 100])
        _ = child1.squeezeChallenge("waste")
        // child2 should still produce the original result
        var ref = parent.fork("child")
        let c2 = child2.squeezeChallenge("out")
        let cRef = ref.squeezeChallenge("out")
        expect(frEq(c2, cRef), "Child mutation does not affect sibling fork")
    }

    // Nested fork
    do {
        var parent = DomainSeparatedTranscript(hash: .poseidon2)
        parent.absorbBytes("root", [1])
        var child = parent.fork("level-1")
        child.absorbBytes("mid", [2])
        var grandchild = child.fork("level-2")
        let gc = grandchild.squeezeChallenge("out")
        let limbs = frToInt(gc)
        let nonZero = limbs[0] != 0 || limbs[1] != 0 || limbs[2] != 0 || limbs[3] != 0
        expect(nonZero, "Nested fork produces non-zero challenge")
    }
}

// MARK: - Cross-hash consistency: all backends produce deterministic output

private func testCrossHashConsistency() {
    let backends: [TranscriptHashType] = [.keccak, .poseidon2, .blake3]

    for backend in backends {
        let name: String
        switch backend {
        case .keccak:   name = "Keccak"
        case .poseidon2: name = "Poseidon2"
        case .blake3:   name = "Blake3"
        }

        // Determinism: same inputs -> same output
        do {
            var t1 = DomainSeparatedTranscript(hash: backend)
            var t2 = DomainSeparatedTranscript(hash: backend)
            t1.beginProtocol("TestProto")
            t2.beginProtocol("TestProto")
            t1.absorbScalar("val", frFromInt(42))
            t2.absorbScalar("val", frFromInt(42))
            t1.absorbBytes("data", [10, 20, 30])
            t2.absorbBytes("data", [10, 20, 30])
            let c1 = t1.squeezeChallenge("alpha")
            let c2 = t2.squeezeChallenge("alpha")
            expect(frEq(c1, c2), "\(name): deterministic output")
        }

        // Non-trivial: challenge is not zero
        do {
            var t = DomainSeparatedTranscript(hash: backend)
            t.absorbBytes("seed", [42])
            let c = t.squeezeChallenge("out")
            let limbs = frToInt(c)
            let nonZero = limbs[0] != 0 || limbs[1] != 0 || limbs[2] != 0 || limbs[3] != 0
            expect(nonZero, "\(name): non-zero challenge")
        }

        // Multiple squeezes produce distinct challenges
        do {
            var t = DomainSeparatedTranscript(hash: backend)
            t.absorbBytes("seed", [1, 2, 3])
            var challenges = [Fr]()
            for i in 0..<5 {
                challenges.append(t.squeezeChallenge("ch_\(i)"))
            }
            var allDistinct = true
            for i in 0..<5 {
                for j in (i + 1)..<5 {
                    if frEq(challenges[i], challenges[j]) {
                        allDistinct = false
                    }
                }
            }
            expect(allDistinct, "\(name): 5 squeezes produce distinct challenges")
        }
    }

    // All three backends produce different outputs from the same input
    do {
        var outputs = [Fr]()
        for backend in backends {
            var t = DomainSeparatedTranscript(hash: backend)
            t.absorbBytes("test", [1, 2, 3])
            outputs.append(t.squeezeChallenge("out"))
        }
        expect(!frEq(outputs[0], outputs[1]), "Keccak != Poseidon2")
        expect(!frEq(outputs[0], outputs[2]), "Keccak != Blake3")
        expect(!frEq(outputs[1], outputs[2]), "Poseidon2 != Blake3")
    }
}

// MARK: - Replay attack: same sequence always produces same challenges

private func testReplayDeterminism() {
    // Full protocol replay produces identical challenges
    do {
        func runProtocol() -> [Fr] {
            var t = DomainSeparatedTranscript(hash: .keccak)
            t.beginProtocol("ReplayTest-v1")
            t.absorbScalar("pub_input", frFromInt(12345))
            t.absorbBytes("witness_hash", [0xAB, 0xCD, 0xEF])
            let alpha = t.squeezeChallenge("alpha")
            t.absorbScalar("round1_sum", frFromInt(67890))
            let beta = t.squeezeChallenge("beta")
            let gamma = t.squeezeChallenge("gamma")
            return [alpha, beta, gamma]
        }

        let run1 = runProtocol()
        let run2 = runProtocol()
        expect(run1.count == 3 && run2.count == 3, "Replay: correct count")
        for i in 0..<3 {
            expect(frEq(run1[i], run2[i]), "Replay: challenge \(i) matches")
        }
    }

    // Replay with Poseidon2
    do {
        func runProtocol() -> Fr {
            var t = DomainSeparatedTranscript(hash: .poseidon2)
            t.beginProtocol("Poseidon2Replay")
            t.absorbBytes("data", [1, 2, 3, 4, 5, 6, 7, 8])
            t.absorbScalar("scalar", frFromInt(999))
            return t.squeezeChallenge("final")
        }
        let c1 = runProtocol()
        let c2 = runProtocol()
        expect(frEq(c1, c2), "Poseidon2 replay determinism")
    }

    // Replay with Blake3
    do {
        func runProtocol() -> Fr {
            var t = DomainSeparatedTranscript(hash: .blake3)
            t.beginProtocol("Blake3Replay")
            for i in 0..<10 {
                t.absorbBytes("step_\(i)", [UInt8(i)])
            }
            return t.squeezeChallenge("result")
        }
        let c1 = runProtocol()
        let c2 = runProtocol()
        expect(frEq(c1, c2), "Blake3 replay determinism")
    }

    // squeezeBytes replay
    do {
        var t1 = DomainSeparatedTranscript(hash: .keccak)
        var t2 = DomainSeparatedTranscript(hash: .keccak)
        t1.absorbBytes("data", [42])
        t2.absorbBytes("data", [42])
        let b1 = t1.squeezeBytes("raw", count: 64)
        let b2 = t2.squeezeBytes("raw", count: 64)
        expect(b1 == b2, "squeezeBytes replay produces identical output")
        expect(b1.count == 64, "squeezeBytes correct length")
    }
}

// MARK: - TranscriptProtocol / Factory tests

private func testTranscriptProtocolFactory() {
    let hashes: [TranscriptHashType] = [.keccak, .poseidon2, .blake3]

    for hash in hashes {
        let name: String
        switch hash {
        case .keccak:   name = "Keccak"
        case .poseidon2: name = "Poseidon2"
        case .blake3:   name = "Blake3"
        }

        // Factory creates working transcript
        do {
            var t = TranscriptFactory.create(hash: hash, label: "factory-test")
            t.appendMessage(label: "data", data: [1, 2, 3])
            let c = t.squeezeChallenge()
            let limbs = frToInt(c)
            let nonZero = limbs[0] != 0 || limbs[1] != 0 || limbs[2] != 0 || limbs[3] != 0
            expect(nonZero, "Factory \(name): produces non-zero challenge")
        }

        // Factory determinism
        do {
            var t1 = TranscriptFactory.create(hash: hash, label: "det-test")
            var t2 = TranscriptFactory.create(hash: hash, label: "det-test")
            t1.appendMessage(label: "msg", data: [7, 8, 9])
            t2.appendMessage(label: "msg", data: [7, 8, 9])
            let c1 = t1.squeezeChallenge()
            let c2 = t2.squeezeChallenge()
            expect(frEq(c1, c2), "Factory \(name): deterministic")
        }

        // Factory hashType is correct
        do {
            let t = TranscriptFactory.create(hash: hash, label: "type-test")
            expect(t.hashType == hash, "Factory \(name): correct hashType")
        }
    }

    // Factory fork works
    do {
        var t = TranscriptFactory.create(hash: .keccak, label: "fork-test")
        t.appendMessage(label: "shared", data: [1, 2, 3])
        var child = t.fork(label: "child")
        let pc = t.squeezeChallenge()
        let cc = child.squeezeChallenge()
        // They should differ because child has additional fork label absorbed
        let limbs = frToInt(pc)
        let nonZero = limbs[0] != 0 || limbs[1] != 0 || limbs[2] != 0 || limbs[3] != 0
        expect(nonZero, "Factory fork: parent produces non-zero")
        let climbs = frToInt(cc)
        let cNonZero = climbs[0] != 0 || climbs[1] != 0 || climbs[2] != 0 || climbs[3] != 0
        expect(cNonZero, "Factory fork: child produces non-zero")
    }

    // TranscriptHashType equality
    do {
        expect(TranscriptHashType.keccak == TranscriptHashType.keccak, "HashType equality: keccak")
        expect(TranscriptHashType.poseidon2 == TranscriptHashType.poseidon2, "HashType equality: poseidon2")
        expect(TranscriptHashType.blake3 == TranscriptHashType.blake3, "HashType equality: blake3")
        expect(TranscriptHashType.keccak != TranscriptHashType.poseidon2, "HashType inequality")
    }
}

// MARK: - Protocol begin tests

private func testProtocolBegin() {
    // Different protocol names produce different challenges
    do {
        var t1 = DomainSeparatedTranscript(hash: .keccak)
        var t2 = DomainSeparatedTranscript(hash: .keccak)
        t1.beginProtocol("Plonk-v1")
        t2.beginProtocol("Groth16")
        t1.absorbBytes("data", [1, 2, 3])
        t2.absorbBytes("data", [1, 2, 3])
        let c1 = t1.squeezeChallenge("alpha")
        let c2 = t2.squeezeChallenge("alpha")
        expect(!frEq(c1, c2), "Different protocol names -> different challenges")
    }

    // Same protocol name, same data -> same challenges
    do {
        var t1 = DomainSeparatedTranscript(hash: .poseidon2)
        var t2 = DomainSeparatedTranscript(hash: .poseidon2)
        t1.beginProtocol("MyProof")
        t2.beginProtocol("MyProof")
        t1.absorbScalar("x", frFromInt(777))
        t2.absorbScalar("x", frFromInt(777))
        let c1 = t1.squeezeChallenge("out")
        let c2 = t2.squeezeChallenge("out")
        expect(frEq(c1, c2), "Same protocol + same data -> same challenges")
    }
}

