// GPUPedersenChainTests -- Tests for GPU Pedersen hash chain engine
// Tests sequential chains, parallel forests, incremental extension,
// chain verification, intermediate verification, vector commitment chains,
// and batch compression.

import zkMetal
import Foundation

public func runGPUPedersenChainTests() {
    suite("GPU Pedersen Chain Engine")

    let engine = GPUPedersenChainEngine(generatorCount: 4)

    // =========================================================================
    // Test 1: Single-message chain equals one Pedersen compression
    // =========================================================================
    do {
        let m0 = frFromInt(42)
        let result = engine.chain(messages: [m0])
        let expected = engine.compress(m0, Fr.zero)  // domainTag = 0
        let expectedHash = engine.pointToFr(expected)
        expect(frEqual(result.finalHash, expectedHash),
               "1-message chain equals H(m_0, 0)")
        expect(result.length == 1, "length == 1")
        expect(result.domainTag == 0, "domainTag == 0")
    }

    // =========================================================================
    // Test 2: Multi-message chain is sequential application
    // =========================================================================
    do {
        let msgs: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30)]
        let result = engine.chain(messages: msgs)

        // Manual recomputation
        let tag = Fr.zero
        let s0 = engine.compress(msgs[0], tag)
        let h0 = engine.pointToFr(s0)
        let s1 = engine.compress(h0, msgs[1])
        let h1 = engine.pointToFr(s1)
        let s2 = engine.compress(h1, msgs[2])
        let h2 = engine.pointToFr(s2)

        expect(frEqual(result.finalHash, h2),
               "3-message chain matches manual recomputation")
        expect(result.length == 3, "length == 3")
    }

    // =========================================================================
    // Test 3: Chain with domain tag differs from chain without
    // =========================================================================
    do {
        let msgs: [Fr] = [frFromInt(99), frFromInt(88)]
        let r0 = engine.chain(messages: msgs, domainTag: 0)
        let r1 = engine.chain(messages: msgs, domainTag: 1)
        expect(!frEqual(r0.finalHash, r1.finalHash),
               "Different domain tags produce different chains")
    }

    // =========================================================================
    // Test 4: Chain determinism
    // =========================================================================
    do {
        let msgs: [Fr] = [frFromInt(7777), frFromInt(8888)]
        let r1 = engine.chain(messages: msgs)
        let r2 = engine.chain(messages: msgs)
        expect(frEqual(r1.finalHash, r2.finalHash),
               "Chain is deterministic")
    }

    // =========================================================================
    // Test 5: Iterative chain (self-hashing)
    // =========================================================================
    do {
        let seed = frFromInt(123)
        let result = engine.iterativeChain(seed: seed, iterations: 5)

        // Manual recomputation
        let tag = Fr.zero
        var h = engine.pointToFr(engine.compress(seed, tag))
        for _ in 1..<5 {
            h = engine.pointToFr(engine.compress(h, tag))
        }
        expect(frEqual(result.finalHash, h),
               "5-iteration iterative chain matches manual")
        expect(result.length == 5, "length == 5")
    }

    // =========================================================================
    // Test 6: Iterative chain with domain tag
    // =========================================================================
    do {
        let seed = frFromInt(555)
        let r0 = engine.iterativeChain(seed: seed, iterations: 3, domainTag: 0)
        let r1 = engine.iterativeChain(seed: seed, iterations: 3, domainTag: 7)
        expect(!frEqual(r0.finalHash, r1.finalHash),
               "Iterative chains with different domain tags differ")
    }

    // =========================================================================
    // Test 7: Parallel forest matches individual chains
    // =========================================================================
    do {
        let msgs1: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let msgs2: [Fr] = [frFromInt(4), frFromInt(5), frFromInt(6)]
        let msgs3: [Fr] = [frFromInt(7), frFromInt(8), frFromInt(9)]

        let forest = engine.chainForest(messageArrays: [msgs1, msgs2, msgs3])

        let r1 = engine.chain(messages: msgs1)
        let r2 = engine.chain(messages: msgs2)
        let r3 = engine.chain(messages: msgs3)

        expect(frEqual(forest.chains[0], r1.finalHash),
               "Forest chain[0] matches individual")
        expect(frEqual(forest.chains[1], r2.finalHash),
               "Forest chain[1] matches individual")
        expect(frEqual(forest.chains[2], r3.finalHash),
               "Forest chain[2] matches individual")
        expect(forest.messagesPerChain == 3, "messagesPerChain == 3")
    }

    // =========================================================================
    // Test 8: Iterative forest matches individual iterative chains
    // =========================================================================
    do {
        let n = 16
        var seeds = [Fr]()
        for i in 0..<n { seeds.append(frFromInt(UInt64(i + 100))) }

        let forest = engine.iterativeForest(seeds: seeds, iterationsPerChain: 7)

        var ok = true
        for i in stride(from: 0, to: n, by: 4) {
            let single = engine.iterativeChain(seed: seeds[i], iterations: 7)
            if !frEqual(forest.chains[i], single.finalHash) {
                ok = false
                break
            }
        }
        expect(ok, "Iterative forest matches individual chains (n=\(n))")
        expect(forest.messagesPerChain == 7, "messagesPerChain == 7")
    }

    // =========================================================================
    // Test 9: Chain extension produces same result as longer chain
    // =========================================================================
    do {
        let msgs: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30),
                           frFromInt(40), frFromInt(50)]

        let full = engine.chain(messages: msgs)

        // Compute first 3, then extend by 2
        let partial = engine.chain(messages: Array(msgs[0..<3]))
        let extended = engine.extendChain(state: partial.finalHash,
                                          messages: Array(msgs[3..<5]))

        expect(frEqual(full.finalHash, extended.finalHash),
               "extend(chain(m0..m2), [m3,m4]) == chain(m0..m4)")
    }

    // =========================================================================
    // Test 10: Iterative chain extension
    // =========================================================================
    do {
        let seed = frFromInt(2025)
        let full = engine.iterativeChain(seed: seed, iterations: 15)

        let partial = engine.iterativeChain(seed: seed, iterations: 10)
        let extended = engine.extendIterativeChain(
            state: partial.finalHash, additionalIterations: 5)

        expect(frEqual(full.finalHash, extended),
               "extendIterativeChain(chain(seed,10), 5) == chain(seed, 15)")
    }

    // =========================================================================
    // Test 11: Iterative forest extension
    // =========================================================================
    do {
        let n = 8
        var seeds = [Fr]()
        for i in 0..<n { seeds.append(frFromInt(UInt64(i + 500))) }

        let full = engine.iterativeForest(seeds: seeds, iterationsPerChain: 12)
        let partial = engine.iterativeForest(seeds: seeds, iterationsPerChain: 7)
        let extended = engine.extendIterativeForest(
            states: partial.chains, additionalIterations: 5)

        var ok = true
        for i in 0..<n {
            if !frEqual(full.chains[i], extended[i]) {
                ok = false
                break
            }
        }
        expect(ok, "extendIterativeForest matches full forest computation")
    }

    // =========================================================================
    // Test 12: Chain verification (correct)
    // =========================================================================
    do {
        let msgs: [Fr] = [frFromInt(11), frFromInt(22), frFromInt(33)]
        let result = engine.chain(messages: msgs)

        let valid = engine.verifyChain(messages: msgs, claimed: result.finalHash)
        expect(valid, "verifyChain accepts correct result")
    }

    // =========================================================================
    // Test 13: Chain verification (incorrect)
    // =========================================================================
    do {
        let msgs: [Fr] = [frFromInt(11), frFromInt(22), frFromInt(33)]

        let invalid = engine.verifyChain(messages: msgs, claimed: frFromInt(999))
        expect(!invalid, "verifyChain rejects incorrect result")
    }

    // =========================================================================
    // Test 14: Iterative chain verification
    // =========================================================================
    do {
        let seed = frFromInt(1234)
        let result = engine.iterativeChain(seed: seed, iterations: 10)

        let valid = engine.verifyIterativeChain(
            seed: seed, iterations: 10, claimed: result.finalHash)
        expect(valid, "verifyIterativeChain accepts correct result")

        let invalid = engine.verifyIterativeChain(
            seed: seed, iterations: 10, claimed: frFromInt(999))
        expect(!invalid, "verifyIterativeChain rejects incorrect result")
    }

    // =========================================================================
    // Test 15: Forest verification
    // =========================================================================
    do {
        let n = 8
        var seeds = [Fr]()
        for i in 0..<n { seeds.append(frFromInt(UInt64(i + 1000))) }

        let forest = engine.iterativeForest(seeds: seeds, iterationsPerChain: 5)
        let results = engine.verifyIterativeForest(
            seeds: seeds, iterationsPerChain: 5, claimed: forest.chains)
        let allValid = results.allSatisfy { $0 }
        expect(allValid, "verifyIterativeForest accepts correct results")

        // Tamper with one chain
        var tampered = forest.chains
        tampered[n / 2] = frFromInt(0)
        let tamperedResults = engine.verifyIterativeForest(
            seeds: seeds, iterationsPerChain: 5, claimed: tampered)
        let tamperedDetected = !tamperedResults[n / 2]
        expect(tamperedDetected, "verifyIterativeForest detects tampered chain")
    }

    // =========================================================================
    // Test 16: Intermediate verification
    // =========================================================================
    do {
        let msgs: [Fr] = [frFromInt(100), frFromInt(200), frFromInt(300)]
        let intermediates = engine.chainIntermediates(messages: msgs)

        expect(intermediates.count == 3, "chainIntermediates returns 3 values")

        let verification = engine.verifyIntermediates(
            messages: msgs, intermediates: intermediates)
        let allOk = verification.allSatisfy { $0 }
        expect(allOk, "verifyIntermediates accepts correct intermediates")

        // Tamper with middle intermediate
        var wrong = intermediates
        wrong[1] = frFromInt(999)
        let wrongVerif = engine.verifyIntermediates(
            messages: msgs, intermediates: wrong)
        expect(wrongVerif[0], "First intermediate still valid after tampering second")
        expect(!wrongVerif[1], "Tampered intermediate detected")
    }

    // =========================================================================
    // Test 17: Compress is non-trivial for non-zero inputs
    // =========================================================================
    do {
        let p = engine.compress(frFromInt(1), frFromInt(0))
        expect(!pointIsIdentity(p), "H(1, 0) is not identity")

        let q = engine.compress(frFromInt(0), frFromInt(1))
        expect(!pointIsIdentity(q), "H(0, 1) is not identity")

        // H(1,0) != H(0,1) (different generators)
        expect(!pointEqual(p, q), "H(1,0) != H(0,1)")
    }

    // =========================================================================
    // Test 18: Zero message produces non-trivial chain (due to domain tag)
    // =========================================================================
    do {
        let result = engine.chain(messages: [Fr.zero], domainTag: 1)
        expect(!frEqual(result.finalHash, Fr.zero),
               "Chain from zero message with domainTag=1 is non-trivial")
    }

    // =========================================================================
    // Test 19: Batch compression matches individual
    // =========================================================================
    do {
        let pairs: [Fr] = [frFromInt(1), frFromInt(2),
                           frFromInt(3), frFromInt(4),
                           frFromInt(5), frFromInt(6)]
        let batch = engine.batchCompress(pairs: pairs)
        expect(batch.count == 3, "batchCompress: 6 elements -> 3 points")

        let p0 = engine.compress(frFromInt(1), frFromInt(2))
        let p1 = engine.compress(frFromInt(3), frFromInt(4))
        let p2 = engine.compress(frFromInt(5), frFromInt(6))

        expect(pointEqual(batch[0], p0), "batch[0] matches individual compress")
        expect(pointEqual(batch[1], p1), "batch[1] matches individual compress")
        expect(pointEqual(batch[2], p2), "batch[2] matches individual compress")
    }

    // =========================================================================
    // Test 20: Batch compress to Fr matches individual
    // =========================================================================
    do {
        let pairs: [Fr] = [frFromInt(10), frFromInt(20),
                           frFromInt(30), frFromInt(40)]
        let frResults = engine.batchCompressToFr(pairs: pairs)
        expect(frResults.count == 2, "batchCompressToFr: 4 elements -> 2 hashes")

        let h0 = engine.pointToFr(engine.compress(frFromInt(10), frFromInt(20)))
        let h1 = engine.pointToFr(engine.compress(frFromInt(30), frFromInt(40)))

        expect(frEqual(frResults[0], h0), "batchCompressToFr[0] matches")
        expect(frEqual(frResults[1], h1), "batchCompressToFr[1] matches")
    }

    // =========================================================================
    // Test 21: hashSingle and hashSingleToFr
    // =========================================================================
    do {
        let m = frFromInt(777)
        let p = engine.hashSingle(m)
        let h = engine.hashSingleToFr(m)
        let expected = engine.compress(m, Fr.zero)
        let expectedH = engine.pointToFr(expected)

        expect(pointEqual(p, expected), "hashSingle matches compress(m, 0)")
        expect(frEqual(h, expectedH), "hashSingleToFr matches pointToFr(compress(m, 0))")
    }

    // =========================================================================
    // Test 22: Vector commitment chain (single step, no chaining)
    // =========================================================================
    do {
        let vec: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let result = engine.vectorCommitChain(vectors: [vec])

        expect(result.steps == 1, "Single-vector chain has 1 step")
        expect(result.intermediates.count == 1, "1 intermediate")
        expect(!pointIsIdentity(result.commitment), "Commitment is non-identity")
    }

    // =========================================================================
    // Test 23: Vector commitment chain (multi-step)
    // =========================================================================
    do {
        let v1: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3)]
        let v2: [Fr] = [frFromInt(4), frFromInt(5), frFromInt(6)]
        let v3: [Fr] = [frFromInt(7), frFromInt(8), frFromInt(9)]

        let result = engine.vectorCommitChain(vectors: [v1, v2, v3])

        expect(result.steps == 3, "3-vector chain has 3 steps")
        expect(result.intermediates.count == 3, "3 intermediates")

        // Each intermediate should be non-identity
        for (i, pt) in result.intermediates.enumerated() {
            expect(!pointIsIdentity(pt),
                   "Intermediate \(i) is non-identity")
        }

        // Verify that changing v2 changes the final commitment
        let v2alt: [Fr] = [frFromInt(4), frFromInt(5), frFromInt(99)]
        let result2 = engine.vectorCommitChain(vectors: [v1, v2alt, v3])
        expect(!pointEqual(result.commitment, result2.commitment),
               "Different vectors produce different chain commitments")
    }

    // =========================================================================
    // Test 24: Vector commitment chain determinism
    // =========================================================================
    do {
        let v1: [Fr] = [frFromInt(10), frFromInt(20)]
        let v2: [Fr] = [frFromInt(30), frFromInt(40)]

        let r1 = engine.vectorCommitChain(vectors: [v1, v2])
        let r2 = engine.vectorCommitChain(vectors: [v1, v2])

        expect(pointEqual(r1.commitment, r2.commitment),
               "Vector commitment chain is deterministic")
    }

    // =========================================================================
    // Test 25: Configurable hash-to-curve (generator derivation determinism)
    // =========================================================================
    do {
        let g0a = GPUPedersenChainEngine.deriveChainGenerator(
            index: 0, seed: "BN254_PedersenChain")
        let g0b = GPUPedersenChainEngine.deriveChainGenerator(
            index: 0, seed: "BN254_PedersenChain")
        let g1 = GPUPedersenChainEngine.deriveChainGenerator(
            index: 1, seed: "BN254_PedersenChain")

        expect(fpToInt(g0a.x) == fpToInt(g0b.x) && fpToInt(g0a.y) == fpToInt(g0b.y),
               "Same index + seed -> same generator")
        expect(fpToInt(g0a.x) != fpToInt(g1.x) || fpToInt(g0a.y) != fpToInt(g1.y),
               "Different indices -> different generators")
    }

    // =========================================================================
    // Test 26: Different seeds produce different generators
    // =========================================================================
    do {
        let gA = GPUPedersenChainEngine.deriveChainGenerator(
            index: 0, seed: "SeedA")
        let gB = GPUPedersenChainEngine.deriveChainGenerator(
            index: 0, seed: "SeedB")

        expect(fpToInt(gA.x) != fpToInt(gB.x) || fpToInt(gA.y) != fpToInt(gB.y),
               "Different seeds -> different generators")
    }

    // =========================================================================
    // Test 27: Engine with different seed produces different chains
    // =========================================================================
    do {
        let eng1 = GPUPedersenChainEngine(generatorCount: 2, seed: "SeedAlpha")
        let eng2 = GPUPedersenChainEngine(generatorCount: 2, seed: "SeedBeta")

        let msgs: [Fr] = [frFromInt(42), frFromInt(43)]
        let r1 = eng1.chain(messages: msgs)
        let r2 = eng2.chain(messages: msgs)

        expect(!frEqual(r1.finalHash, r2.finalHash),
               "Engines with different seeds produce different chains")
    }

    // =========================================================================
    // Test 28: Chain intermediates length matches messages
    // =========================================================================
    do {
        let msgs: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3),
                           frFromInt(4), frFromInt(5)]
        let intermediates = engine.chainIntermediates(messages: msgs)

        expect(intermediates.count == 5, "chainIntermediates count == 5")

        // Last intermediate should equal chain final hash
        let result = engine.chain(messages: msgs)
        expect(frEqual(intermediates[4], result.finalHash),
               "Last intermediate equals chain final hash")
    }

    // =========================================================================
    // Test 29: Performance benchmark -- iterative chain
    // =========================================================================
    do {
        let seed = frFromInt(12345)
        let iters = 100

        let t0 = CFAbsoluteTimeGetCurrent()
        let result = engine.iterativeChain(seed: seed, iterations: iters)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        expect(!frEqual(result.finalHash, Fr.zero),
               String(format: "Iterative chain %d iters: %.1fms (%.2f us/iter)",
                      iters, elapsed * 1000,
                      elapsed / Double(iters) * 1e6))
    }

    // =========================================================================
    // Test 30: Performance benchmark -- parallel forest
    // =========================================================================
    do {
        let n = 64
        var seeds = [Fr]()
        seeds.reserveCapacity(n)
        var lcg: UInt64 = 0xDEAD_BEEF
        for _ in 0..<n {
            lcg = lcg &* 6364136223846793005 &+ 1442695040888963407
            seeds.append(frFromInt(lcg >> 32))
        }

        let iters = 20
        let t0 = CFAbsoluteTimeGetCurrent()
        let forest = engine.iterativeForest(seeds: seeds, iterationsPerChain: iters)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        expect(forest.chains.count == n,
               String(format: "Forest %dx%d: %.1fms (%.2f us/chain-iter)",
                      n, iters, elapsed * 1000,
                      elapsed / Double(n * iters) * 1e6))
    }
}
