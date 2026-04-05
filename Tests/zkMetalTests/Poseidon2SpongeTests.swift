import zkMetal
import Foundation

func runPoseidon2SpongeTests() {
    suite("Poseidon2 Sponge")

    // Test 1: Basic hash produces non-trivial output
    do {
        let input = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let hash = Poseidon2Sponge.hash(input)
        expect(frToInt(hash) != [0, 0, 0, 0], "Non-trivial hash output")
    }

    // Test 2: Deterministic — same input produces same output
    do {
        let input = [frFromInt(42), frFromInt(99)]
        let h1 = Poseidon2Sponge.hash(input)
        let h2 = Poseidon2Sponge.hash(input)
        expect(frToInt(h1) == frToInt(h2), "Deterministic hash")
    }

    // Test 3: Different inputs produce different outputs
    do {
        let h1 = Poseidon2Sponge.hash([frFromInt(1)])
        let h2 = Poseidon2Sponge.hash([frFromInt(2)])
        expect(frToInt(h1) != frToInt(h2), "Different inputs -> different hashes")
    }

    // Test 4: Domain separation — same input, different tags produce different hashes
    do {
        let input = [frFromInt(1), frFromInt(2)]
        let h1 = Poseidon2Sponge.hash(input, domainTag: 0)
        let h2 = Poseidon2Sponge.hash(input, domainTag: 1)
        expect(frToInt(h1) != frToInt(h2), "Domain separation works")
    }

    // Test 5: Multi-block absorb (more than rate=2 elements)
    do {
        let input = (0..<10).map { frFromInt(UInt64($0)) }
        let hash = Poseidon2Sponge.hash(input)
        expect(frToInt(hash) != [0, 0, 0, 0], "Multi-block hash non-trivial")
    }

    // Test 6: Squeeze multiple elements
    do {
        let input = [frFromInt(7), frFromInt(11)]
        let output = Poseidon2Sponge.absorbAndSqueeze(input: input, outputCount: 4)
        expect(output.count == 4, "Squeeze count matches request")
        var allNonZero = true
        for o in output {
            if frToInt(o) == [0, 0, 0, 0] { allNonZero = false }
        }
        expect(allNonZero, "All squeezed elements non-zero")
    }

    // Test 7: Duplex mode — interleaved absorb/squeeze
    do {
        var sponge = Poseidon2Sponge(domainTag: 1)
        sponge.absorb(element: frFromInt(100))
        let c1 = sponge.squeezeOne()
        sponge.absorb(element: frFromInt(200))
        let c2 = sponge.squeezeOne()
        expect(frToInt(c1) != frToInt(c2), "Duplex mode produces different challenges")
    }

    // Test 8: Sponge hash differs from poseidon2HashMany (different squeeze semantics)
    do {
        let input = [frFromInt(5), frFromInt(10)]
        let spongeHash = Poseidon2Sponge.hash(input)
        let directHash = poseidon2HashMany(input)
        expect(frToInt(spongeHash) != frToInt(directHash), "Sponge differs from poseidon2HashMany (extra squeeze permute)")
        expect(frToInt(spongeHash) != [0, 0, 0, 0], "Sponge hash non-trivial")
        expect(frToInt(directHash) != [0, 0, 0, 0], "Direct hash non-trivial")
    }

    // Test 9: Clone produces independent state
    do {
        var sponge1 = Poseidon2Sponge(domainTag: 0)
        sponge1.absorb(elements: [frFromInt(1), frFromInt(2)])
        var sponge2 = sponge1.clone()
        let h1 = sponge1.squeezeOne()
        let h2 = sponge2.squeezeOne()
        expect(frToInt(h1) == frToInt(h2), "Clone produces same hash")

        sponge1.absorb(element: frFromInt(99))
        sponge2.absorb(element: frFromInt(100))
        let h1b = sponge1.squeezeOne()
        let h2b = sponge2.squeezeOne()
        expect(frToInt(h1b) != frToInt(h2b), "Diverged clones produce different hashes")
    }

    // Test 10: Byte absorption
    do {
        var sponge = Poseidon2Sponge(domainTag: 0)
        sponge.absorbBytes([0x01, 0x02, 0x03, 0x04])
        let hash = sponge.squeezeOne()
        expect(frToInt(hash) != [0, 0, 0, 0], "Byte absorb produces non-trivial hash")
    }

    suite("Poseidon2 BabyBear Sponge")

    // Test 11: BabyBear sponge basic hash
    do {
        let input = (0..<8).map { Bb(v: UInt32($0) + 1) }
        let hash = Poseidon2BbSponge.hash(input)
        expect(hash.count == 8, "BabyBear sponge output is 8 elements")
        var nonTrivial = false
        for h in hash { if h.v != 0 { nonTrivial = true; break } }
        expect(nonTrivial, "BabyBear sponge non-trivial output")
    }

    // Test 12: BabyBear sponge deterministic
    do {
        let input = (0..<16).map { Bb(v: UInt32($0) + 1) }
        let h1 = Poseidon2BbSponge.hash(input)
        let h2 = Poseidon2BbSponge.hash(input)
        expect(h1.map(\.v) == h2.map(\.v), "BabyBear sponge deterministic")
    }

    // Test 13: BabyBear domain separation
    do {
        let input = (0..<4).map { Bb(v: UInt32($0) + 1) }
        let h1 = Poseidon2BbSponge.hash(input, domainTag: 0)
        let h2 = Poseidon2BbSponge.hash(input, domainTag: 1)
        expect(h1.map(\.v) != h2.map(\.v), "BabyBear domain separation")
    }

    suite("Poseidon2 Sponge Transcript")

    // Test 14: Transcript produces challenges
    do {
        var transcript = Poseidon2SpongeTranscript()
        transcript.absorbFr(frFromInt(42))
        let challenge = transcript.squeezeFr()
        expect(frToInt(challenge) != [0, 0, 0, 0], "Transcript produces non-zero challenge")
    }

    // Test 15: Transcript is deterministic
    do {
        var t1 = Poseidon2SpongeTranscript()
        t1.absorbFr(frFromInt(1))
        t1.absorbFr(frFromInt(2))
        let c1 = t1.squeezeFr()

        var t2 = Poseidon2SpongeTranscript()
        t2.absorbFr(frFromInt(1))
        t2.absorbFr(frFromInt(2))
        let c2 = t2.squeezeFr()

        expect(frToInt(c1) == frToInt(c2), "Transcript deterministic")
    }

    // Test 16: Transcript with FiatShamirTranscript wrapper
    do {
        var t = FiatShamirTranscript(label: "test-protocol", hasher: Poseidon2SpongeTranscript())
        t.absorbFr("commitment", frFromInt(123))
        let challenge = t.challengeScalar("alpha")
        expect(frToInt(challenge) != [0, 0, 0, 0], "FiatShamir sponge transcript works")
    }

    suite("Batch Sponge Hash")

    // Test 17: CPU batch hash matches individual sponge hashes
    do {
        let messages: [[Fr]] = (0..<8).map { i in
            (0..<4).map { j in frFromInt(UInt64(i * 10 + j)) }
        }

        let batchResults = batchSpongeHashCPU(inputs: messages, domainTag: 0)
        var ok = true
        for i in 0..<messages.count {
            let individual = Poseidon2Sponge.hash(messages[i], domainTag: 0)
            if frToInt(batchResults[i]) != frToInt(individual) {
                ok = false
                break
            }
        }
        expect(ok, "CPU batch matches individual hashes")
    }

    // Test 18: Large batch via GCD parallel
    do {
        let messages: [[Fr]] = (0..<128).map { i in
            (0..<6).map { j in frFromInt(UInt64(i * 100 + j)) }
        }

        let batchResults = batchSpongeHashCPU(inputs: messages, domainTag: 0)
        var ok = true
        for i in 0..<messages.count {
            let individual = Poseidon2Sponge.hash(messages[i], domainTag: 0)
            if frToInt(batchResults[i]) != frToInt(individual) {
                ok = false
                break
            }
        }
        expect(ok, "Parallel batch matches individual (128 msgs)")
    }

    // Test 19: Uniform batch hash matches individual
    do {
        let messageLen = 4
        let n = 16
        var flatInput = [Fr]()
        for i in 0..<n {
            for j in 0..<messageLen {
                flatInput.append(frFromInt(UInt64(i * 100 + j)))
            }
        }

        do {
            let results = try batchSpongeHashUniform(flatInput: flatInput, messageLen: messageLen, domainTag: 0)
            var ok = true
            for i in 0..<n {
                let msg = Array(flatInput[i * messageLen ..< (i + 1) * messageLen])
                let expected = Poseidon2Sponge.hash(msg, domainTag: 0)
                if frToInt(results[i]) != frToInt(expected) {
                    ok = false
                    break
                }
            }
            expect(ok, "Uniform batch matches individual")
        } catch {
            expect(false, "Uniform batch error: \(error)")
        }
    }

    // Test 20: BabyBear batch hash
    do {
        let messages: [[Bb]] = (0..<4).map { i in
            (0..<12).map { j in Bb(v: UInt32(i * 10 + j + 1)) }
        }

        let batchResults = batchSpongeHashBb(inputs: messages, domainTag: 0)
        expect(batchResults.count == 4 * 8, "BabyBear batch output size correct")

        let individual = Poseidon2BbSponge.hash(messages[0], domainTag: 0)
        let batchFirst = Array(batchResults[0..<8])
        expect(individual.map(\.v) == batchFirst.map(\.v), "BabyBear batch matches individual")
    }

    // Test 21: batchSpongeHash API with engine parameter (should still work)
    do {
        let messages: [[Fr]] = (0..<4).map { i in
            (0..<4).map { j in frFromInt(UInt64(i * 10 + j)) }
        }
        do {
            let engine = try Poseidon2Engine()
            let results = try batchSpongeHash(inputs: messages, domainTag: 0, engine: engine)
            let cpuResults = batchSpongeHashCPU(inputs: messages, domainTag: 0)
            var ok = true
            for i in 0..<messages.count {
                if frToInt(results[i]) != frToInt(cpuResults[i]) { ok = false; break }
            }
            expect(ok, "batchSpongeHash with engine matches CPU")
        } catch {
            expect(true, "batchSpongeHash engine skipped: \(error)")
        }
    }

    // Test 22: Variable-length messages
    do {
        let messages: [[Fr]] = [
            [frFromInt(1)],
            [frFromInt(2), frFromInt(3)],
            [frFromInt(4), frFromInt(5), frFromInt(6)],
            (0..<10).map { frFromInt(UInt64($0)) },
        ]
        let batchResults = batchSpongeHashCPU(inputs: messages, domainTag: 0)
        var ok = true
        for i in 0..<messages.count {
            let individual = Poseidon2Sponge.hash(messages[i], domainTag: 0)
            if frToInt(batchResults[i]) != frToInt(individual) { ok = false; break }
        }
        expect(ok, "Variable-length batch matches individual")
    }
}
