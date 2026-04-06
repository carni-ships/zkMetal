// GPUPoseidon2ChainTests — Tests for GPU Poseidon2 hash chain engine
// Tests sequential chains, parallel forests, domain separation, Merkle trees,
// incremental extension, and verification.

import zkMetal
import Foundation

public func runGPUPoseidon2ChainTests() {
    suite("GPU Poseidon2 Chain Engine")

    guard let engine = try? GPUPoseidon2ChainEngine() else {
        print("  [SKIP] No GPU available")
        return
    }

    // =========================================================================
    // Test 1: Single-iteration chain equals one Poseidon2 hash
    // =========================================================================
    do {
        let seed = frFromInt(42)
        let result = try engine.chain(seed: seed, iterations: 1)
        let expected = poseidon2Hash(seed, Fr.zero)
        expect(frEqual(result.finalHash, expected),
               "1-iteration chain equals H(seed, 0)")
        expect(result.iterations == 1, "iterations == 1")
        expect(result.domainTag == 0, "domainTag == 0")
    } catch {
        expect(false, "Single-iteration chain error: \(error)")
    }

    // =========================================================================
    // Test 2: Multi-iteration chain is sequential application
    // =========================================================================
    do {
        let seed = frFromInt(123)
        let n = 10
        let result = try engine.chain(seed: seed, iterations: n)

        // Recompute manually
        var manual = seed
        for _ in 0..<n {
            manual = poseidon2Hash(manual, Fr.zero)
        }
        expect(frEqual(result.finalHash, manual),
               "\(n)-iteration chain matches manual recomputation")
    } catch {
        expect(false, "Multi-iteration chain error: \(error)")
    }

    // =========================================================================
    // Test 3: Chain with domain tag differs from chain without
    // =========================================================================
    do {
        let seed = frFromInt(99)
        let r0 = try engine.chain(seed: seed, iterations: 5, domainTag: 0)
        let r1 = try engine.chain(seed: seed, iterations: 5, domainTag: 1)
        expect(!frEqual(r0.finalHash, r1.finalHash),
               "Different domain tags produce different chains")
    } catch {
        expect(false, "Domain tag chain error: \(error)")
    }

    // =========================================================================
    // Test 4: Chain determinism
    // =========================================================================
    do {
        let seed = frFromInt(7777)
        let r1 = try engine.chain(seed: seed, iterations: 20)
        let r2 = try engine.chain(seed: seed, iterations: 20)
        expect(frEqual(r1.finalHash, r2.finalHash),
               "Chain is deterministic")
    } catch {
        expect(false, "Determinism error: \(error)")
    }

    // =========================================================================
    // Test 5: Parallel forest matches sequential chains (small, CPU path)
    // =========================================================================
    do {
        let n = 8  // below gpuForestThreshold
        var seeds = [Fr]()
        for i in 0..<n { seeds.append(frFromInt(UInt64(i + 1))) }

        let forest = try engine.chainForest(seeds: seeds, iterationsPerChain: 5)

        // Verify each chain independently
        var ok = true
        for i in 0..<n {
            let single = try engine.chain(seed: seeds[i], iterations: 5)
            if !frEqual(forest.chains[i], single.finalHash) {
                ok = false
                break
            }
        }
        expect(ok, "Small forest matches individual chains (n=\(n))")
    } catch {
        expect(false, "Small forest error: \(error)")
    }

    // =========================================================================
    // Test 6: Parallel forest matches sequential chains (large, GPU path)
    // =========================================================================
    do {
        let n = 64  // above gpuForestThreshold
        var seeds = [Fr]()
        for i in 0..<n { seeds.append(frFromInt(UInt64(i + 100))) }

        let forest = try engine.chainForest(seeds: seeds, iterationsPerChain: 10)

        // Verify a sample of chains
        var ok = true
        for i in stride(from: 0, to: n, by: 8) {
            let single = try engine.chain(seed: seeds[i], iterations: 10)
            if !frEqual(forest.chains[i], single.finalHash) {
                ok = false
                break
            }
        }
        expect(ok, "GPU forest matches individual chains (n=\(n))")
        expect(forest.iterationsPerChain == 10, "iterationsPerChain == 10")
    } catch {
        expect(false, "GPU forest error: \(error)")
    }

    // =========================================================================
    // Test 7: Domain-separated chain differs from plain chain
    // =========================================================================
    do {
        let seed = frFromInt(555)
        let plain = try engine.chain(seed: seed, iterations: 5, domainTag: 7)
        let domSep = try engine.domainSeparatedChain(seed: seed, iterations: 5, domainTag: 7)
        // Domain-separated chain XORs iteration index into tag, so should differ
        expect(!frEqual(plain.finalHash, domSep.finalHash),
               "Domain-separated chain differs from plain chain")
    } catch {
        expect(false, "Domain-separated chain error: \(error)")
    }

    // =========================================================================
    // Test 8: Domain-separated chain is deterministic and correct
    // =========================================================================
    do {
        let seed = frFromInt(333)
        let r1 = try engine.domainSeparatedChain(seed: seed, iterations: 5, domainTag: 42)
        let r2 = try engine.domainSeparatedChain(seed: seed, iterations: 5, domainTag: 42)
        expect(frEqual(r1.finalHash, r2.finalHash),
               "Domain-separated chain is deterministic")

        // Manual verification
        var manual = seed
        for i in 0..<5 {
            let tag = frFromInt(42 ^ UInt64(i))
            manual = poseidon2Hash(manual, tag)
        }
        expect(frEqual(r1.finalHash, manual),
               "Domain-separated chain matches manual computation")
    } catch {
        expect(false, "Domain-separated determinism error: \(error)")
    }

    // =========================================================================
    // Test 9: Domain-separated forest matches individual chains
    // =========================================================================
    do {
        let n = 32
        var seeds = [Fr]()
        for i in 0..<n { seeds.append(frFromInt(UInt64(i * 7 + 13))) }

        let forest = try engine.domainSeparatedForest(seeds: seeds, iterationsPerChain: 8,
                                                       domainTag: 99)

        var ok = true
        for i in stride(from: 0, to: n, by: 4) {
            let single = try engine.domainSeparatedChain(seed: seeds[i], iterations: 8,
                                                          domainTag: 99)
            if !frEqual(forest.chains[i], single.finalHash) {
                ok = false
                break
            }
        }
        expect(ok, "Domain-separated forest matches individual chains (n=\(n))")
    } catch {
        expect(false, "Domain-separated forest error: \(error)")
    }

    // =========================================================================
    // Test 10: Chain extension produces same result as longer chain
    // =========================================================================
    do {
        let seed = frFromInt(2025)
        let full = try engine.chain(seed: seed, iterations: 15)

        // Compute first 10, then extend by 5
        let partial = try engine.chain(seed: seed, iterations: 10)
        let extended = try engine.extendChain(state: partial.finalHash, additionalIterations: 5)

        expect(frEqual(full.finalHash, extended),
               "extend(chain(seed,10), 5) == chain(seed, 15)")
    } catch {
        expect(false, "Chain extension error: \(error)")
    }

    // =========================================================================
    // Test 11: Forest extension
    // =========================================================================
    do {
        let n = 32
        var seeds = [Fr]()
        for i in 0..<n { seeds.append(frFromInt(UInt64(i + 500))) }

        let full = try engine.chainForest(seeds: seeds, iterationsPerChain: 12)
        let partial = try engine.chainForest(seeds: seeds, iterationsPerChain: 7)
        let extended = try engine.extendForest(states: partial.chains, additionalIterations: 5)

        var ok = true
        for i in 0..<n {
            if !frEqual(full.chains[i], extended[i]) {
                ok = false
                break
            }
        }
        expect(ok, "extendForest matches full forest computation")
    } catch {
        expect(false, "Forest extension error: \(error)")
    }

    // =========================================================================
    // Test 12: Chain verification
    // =========================================================================
    do {
        let seed = frFromInt(1234)
        let result = try engine.chain(seed: seed, iterations: 10)

        let valid = try engine.verifyChain(seed: seed, iterations: 10,
                                            claimed: result.finalHash)
        expect(valid, "verifyChain accepts correct result")

        let invalid = try engine.verifyChain(seed: seed, iterations: 10,
                                              claimed: frFromInt(999))
        expect(!invalid, "verifyChain rejects incorrect result")
    } catch {
        expect(false, "Chain verification error: \(error)")
    }

    // =========================================================================
    // Test 13: Forest verification
    // =========================================================================
    do {
        let n = 32
        var seeds = [Fr]()
        for i in 0..<n { seeds.append(frFromInt(UInt64(i + 1000))) }

        let forest = try engine.chainForest(seeds: seeds, iterationsPerChain: 5)
        let results = try engine.verifyForest(seeds: seeds, iterationsPerChain: 5,
                                               claimed: forest.chains)
        let allValid = results.allSatisfy { $0 }
        expect(allValid, "verifyForest accepts correct results")

        // Tamper with one chain
        var tampered = forest.chains
        tampered[n / 2] = frFromInt(0)
        let tamperedResults = try engine.verifyForest(seeds: seeds, iterationsPerChain: 5,
                                                       claimed: tampered)
        let tamperedDetected = !tamperedResults[n / 2]
        expect(tamperedDetected, "verifyForest detects tampered chain")
    } catch {
        expect(false, "Forest verification error: \(error)")
    }

    // =========================================================================
    // Test 14: Merkle tree from chain endpoints
    // =========================================================================
    do {
        let n = 16  // power of 2
        var seeds = [Fr]()
        for i in 0..<n { seeds.append(frFromInt(UInt64(i + 2000))) }

        let tree = try engine.merkleTreeFromChains(seeds: seeds, iterationsPerChain: 5)
        expect(tree.leafCount == n, "Merkle tree has \(n) leaves")
        expect(tree.depth == 4, "Merkle tree depth == 4")

        // Verify a proof
        let proof = tree.proof(forLeafAt: 3)
        let valid = proof.verify(root: tree.root, leaf: tree.leaf(at: 3))
        expect(valid, "Merkle proof from chain tree verifies")

        // Root from chains should match
        let root = try engine.merkleRootFromChains(seeds: seeds, iterationsPerChain: 5)
        expect(frEqual(root, tree.root), "merkleRootFromChains matches buildTree root")
    } catch {
        expect(false, "Merkle tree from chains error: \(error)")
    }

    // =========================================================================
    // Test 15: Merkle tree from data pairs
    // =========================================================================
    do {
        let n = 8
        var data = [(Fr, Fr)]()
        for i in 0..<n {
            data.append((frFromInt(UInt64(i * 2)), frFromInt(UInt64(i * 2 + 1))))
        }

        let tree = try engine.merkleTreeFromPairs(data: data)
        expect(tree.leafCount == n, "Pair tree has \(n) leaves")

        // Leaf should be H(data[i].0, data[i].1)
        let expectedLeaf0 = poseidon2Hash(data[0].0, data[0].1)
        expect(frEqual(tree.leaf(at: 0), expectedLeaf0),
               "Leaf 0 matches H(pair)")

        // Proof verification
        let proof = tree.proof(forLeafAt: 5)
        let valid = proof.verify(root: tree.root, leaf: tree.leaf(at: 5))
        expect(valid, "Merkle proof from pair tree verifies")
    } catch {
        expect(false, "Merkle tree from pairs error: \(error)")
    }

    // =========================================================================
    // Test 16: hashPairsToLeaves
    // =========================================================================
    do {
        let data: [Fr] = [frFromInt(10), frFromInt(20), frFromInt(30), frFromInt(40)]
        let leaves = try engine.hashPairsToLeaves(data)
        expect(leaves.count == 2, "hashPairsToLeaves: 4 elements -> 2 leaves")
        let expected0 = poseidon2Hash(frFromInt(10), frFromInt(20))
        let expected1 = poseidon2Hash(frFromInt(30), frFromInt(40))
        expect(frEqual(leaves[0], expected0), "leaf[0] matches manual hash")
        expect(frEqual(leaves[1], expected1), "leaf[1] matches manual hash")
    } catch {
        expect(false, "hashPairsToLeaves error: \(error)")
    }

    // =========================================================================
    // Test 17: hashBlocks
    // =========================================================================
    do {
        let blocks: [[Fr]] = [
            [frFromInt(1), frFromInt(2), frFromInt(3)],
            [frFromInt(4), frFromInt(5)],
            [frFromInt(6)],
        ]
        let results = try engine.hashBlocks(blocks)
        expect(results.count == 3, "hashBlocks: 3 blocks -> 3 hashes")

        let expected0 = poseidon2HashMany([frFromInt(1), frFromInt(2), frFromInt(3)])
        expect(frEqual(results[0], expected0), "hashBlocks[0] matches poseidon2HashMany")
    } catch {
        expect(false, "hashBlocks error: \(error)")
    }

    // =========================================================================
    // Test 18: Large forest performance benchmark
    // =========================================================================
    do {
        let n = 1024
        var seeds = [Fr]()
        seeds.reserveCapacity(n)
        var lcg: UInt64 = 0xDEAD_BEEF
        for _ in 0..<n {
            lcg = lcg &* 6364136223846793005 &+ 1442695040888963407
            seeds.append(frFromInt(lcg >> 32))
        }

        let iters = 50
        let t0 = CFAbsoluteTimeGetCurrent()
        let forest = try engine.chainForest(seeds: seeds, iterationsPerChain: iters)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        expect(forest.chains.count == n,
               String(format: "Forest %dx%d: %.1fms (%.2f us/chain-iter)",
                      n, iters, elapsed * 1000,
                      elapsed / Double(n * iters) * 1e6))
    } catch {
        expect(false, "Forest benchmark error: \(error)")
    }

    // =========================================================================
    // Test 19: Zero seed chain produces non-trivial output
    // =========================================================================
    do {
        let result = try engine.chain(seed: Fr.zero, iterations: 1)
        // H(0, 0) should not be zero (Poseidon2 is a permutation, not identity)
        expect(!frEqual(result.finalHash, Fr.zero),
               "Chain from zero seed is non-trivial")
    } catch {
        expect(false, "Zero seed chain error: \(error)")
    }
}
