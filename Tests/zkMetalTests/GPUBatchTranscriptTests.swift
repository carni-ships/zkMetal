import zkMetal
import Foundation

/// Compare Fr elements for equality.
private func frMatch(_ a: Fr, _ b: Fr) -> Bool {
    return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3
        && a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

func runGPUBatchTranscriptTests() {
    suite("GPUBatchTranscript")

    guard let engine = try? GPUBatchTranscript() else {
        expect(false, "GPU not available, skipping batch transcript tests")
        return
    }

    // Helper: generate deterministic test elements
    func genElems(_ count: Int, seed: UInt64 = 0xDEAD_BEEF) -> [Fr] {
        var elems = [Fr]()
        var rng = seed
        for _ in 0..<count {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            elems.append(frFromInt(rng))
        }
        return elems
    }

    let elems = genElems(2048)

    // Test 1: GPU determinism
    do {
        let n = 128
        let messages = (0..<n).map { i in Array(elems[i * 4 ..< (i + 1) * 4]) }
        let r1 = try engine.batchAbsorb(values: messages, domainTag: 0)
        let r2 = try engine.batchAbsorb(values: messages, domainTag: 0)
        var match = true
        for i in 0..<n { if !frMatch(r1[i], r2[i]) { match = false; break } }
        expect(match, "GPU determinism: same inputs -> same outputs")
    } catch {
        expect(false, "GPU determinism threw: \(error)")
    }

    // Test 2: Domain separation
    do {
        let n = 128
        let messages = (0..<n).map { i in Array(elems[i * 4 ..< (i + 1) * 4]) }
        let r1 = try engine.batchAbsorb(values: messages, domainTag: 0)
        let r2 = try engine.batchAbsorb(values: messages, domainTag: 1)
        var allDiff = true
        for i in 0..<n { if frMatch(r1[i], r2[i]) { allDiff = false; break } }
        expect(allDiff, "domain separation: different tags -> different outputs")
    } catch {
        expect(false, "domain separation threw: \(error)")
    }

    // Test 3: Different inputs -> different outputs
    do {
        let n = 128
        let messages = (0..<n).map { i in Array(elems[i * 4 ..< (i + 1) * 4]) }
        let results = try engine.batchAbsorb(values: messages, domainTag: 0)
        var allDiff = true
        for i in 0..<(n - 1) { if frMatch(results[i], results[i + 1]) { allDiff = false; break } }
        expect(allDiff, "different inputs -> different outputs")
    } catch {
        expect(false, "different inputs threw: \(error)")
    }

    // Test 4: Multi-squeeze distinct
    do {
        let n = 64
        let messages = (0..<n).map { i in Array(elems[i * 4 ..< (i + 1) * 4]) }
        let states = engine.initStates(count: n, domainTag: 0)
        let after = try engine.absorbUniform(states: states, messages: messages)
        let (_, sq) = try engine.batchSqueeze(states: after, count: 3)
        var distinct = true
        for i in 0..<n {
            if frMatch(sq[i][0], sq[i][1]) || frMatch(sq[i][0], sq[i][2]) || frMatch(sq[i][1], sq[i][2]) {
                distinct = false; break
            }
        }
        expect(distinct, "multi-squeeze produces distinct challenges")
    } catch {
        expect(false, "multi-squeeze threw: \(error)")
    }

    // Test 5: CPU fallback matches CPU sponge
    do {
        let n = 8  // well below threshold
        let messages = (0..<n).map { i in Array(elems[i * 4 ..< (i + 1) * 4]) }
        let results = try engine.batchAbsorb(values: messages, domainTag: 0)
        var match = true
        for i in 0..<n {
            // CPU reference
            var s0 = Fr.zero, s1 = Fr.zero, s2 = Fr.zero
            var abs = 0
            for elem in messages[i] {
                if abs == 0 { s0 = frAdd(s0, elem) } else { s1 = frAdd(s1, elem) }
                abs += 1
                if abs == 2 { poseidon2PermuteInPlace(&s0, &s1, &s2); abs = 0 }
            }
            poseidon2PermuteInPlace(&s0, &s1, &s2)
            if !frMatch(results[i], s0) { match = false; break }
        }
        expect(match, "CPU fallback matches CPU sponge (n=\(n))")
    } catch {
        expect(false, "CPU fallback threw: \(error)")
    }

    // Test 6: Variable-length absorb determinism
    do {
        let n = 128
        var messages = [[Fr]]()
        for i in 0..<n {
            let len = (i % 5) + 1
            let start = (i * 5) % (elems.count - 5)
            messages.append(Array(elems[start ..< start + len]))
        }
        let states = engine.initStates(count: n, domainTag: 3)
        let a1 = try engine.absorbVarlen(states: states, messages: messages)
        let (_, sq1) = try engine.batchSqueeze(states: a1, count: 1)

        let states2 = engine.initStates(count: n, domainTag: 3)
        let a2 = try engine.absorbVarlen(states: states2, messages: messages)
        let (_, sq2) = try engine.batchSqueeze(states: a2, count: 1)

        var match = true
        for i in 0..<n { if !frMatch(sq1[i][0], sq2[i][0]) { match = false; break } }
        expect(match, "varlen absorb determinism")
    } catch {
        expect(false, "varlen absorb threw: \(error)")
    }

    // Test 7: Non-zero output
    do {
        let n = 64
        let messages = (0..<n).map { i in Array(elems[i * 4 ..< (i + 1) * 4]) }
        let results = try engine.batchAbsorb(values: messages, domainTag: 0)
        var anyNonZero = false
        for r in results { if !frMatch(r, Fr.zero) { anyNonZero = true; break } }
        expect(anyNonZero, "non-zero GPU output")
    } catch {
        expect(false, "non-zero check threw: \(error)")
    }

    // Test 8: Multi-step pipeline determinism
    do {
        let n = 64
        let m1 = (0..<n).map { i in Array(elems[i * 4 ..< (i + 1) * 4]) }
        let m2 = (0..<n).map { i in Array(elems[(n + i) * 4 ..< (n + i + 1) * 4]) }

        var s = engine.initStates(count: n, domainTag: 9)
        s = try engine.absorbUniform(states: s, messages: m1)
        let (s2, sq1) = try engine.batchSqueeze(states: s, count: 1)
        let s3 = try engine.absorbUniform(states: s2, messages: m2)
        let (_, sq2) = try engine.batchSqueeze(states: s3, count: 1)

        // Run again
        var sB = engine.initStates(count: n, domainTag: 9)
        sB = try engine.absorbUniform(states: sB, messages: m1)
        let (s2B, sq1B) = try engine.batchSqueeze(states: sB, count: 1)
        let s3B = try engine.absorbUniform(states: s2B, messages: m2)
        let (_, sq2B) = try engine.batchSqueeze(states: s3B, count: 1)

        var match = true
        for i in 0..<n {
            if !frMatch(sq1[i][0], sq1B[i][0]) || !frMatch(sq2[i][0], sq2B[i][0]) {
                match = false; break
            }
        }
        expect(match, "multi-step pipeline determinism")
    } catch {
        expect(false, "multi-step threw: \(error)")
    }

    // Test 9: Empty message absorb
    do {
        let n = 64
        let messages = [[Fr]](repeating: [], count: n)
        let states = engine.initStates(count: n, domainTag: 0)
        let after = try engine.absorbUniform(states: states, messages: messages)
        let (_, sq) = try engine.batchSqueeze(states: after, count: 1)
        // All should be the same (same domain tag, empty message)
        var allSame = true
        for i in 1..<n { if !frMatch(sq[0][0], sq[i][0]) { allSame = false; break } }
        expect(allSame, "empty message: all transcripts produce same output")
    } catch {
        expect(false, "empty message threw: \(error)")
    }
}
