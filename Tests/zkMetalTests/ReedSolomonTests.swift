// Reed-Solomon Engine Tests
// Tests for ReedSolomonEngine (BN254 Fr) and ReedSolomonBbEngine (BabyBear):
// - Encode/decode round-trip
// - Decode from random k-of-n subsets (multiple erasure patterns)
// - Verify valid and invalid codewords
// - Multi-block erasure coding round-trip
// - NTT-based encode matches naive polynomial evaluation
// - Performance benchmark

import Foundation
import zkMetal

public func runReedSolomonEngineTests() {
    suite("Reed-Solomon Engine")

    // ------------------------------------------------------------------
    // 1. BN254 Fr: Encode then decode recovers original data
    // ------------------------------------------------------------------
    do {
        let engine = try! ReedSolomonEngine()
        let k = 8
        var data = [Fr]()
        for i in 0..<k {
            data.append(frFromInt(UInt64(i + 1)))
        }

        let codeword = try! engine.encode(data: data, codeRate: 2)
        let n = codeword.count
        expect(n == 16, "BN254: codeword length = 2 * nextPow2(8) = 16")

        // Decode from first k indices
        let pairs: [(Int, Fr)] = (0..<k).map { i in (i, codeword[i]) }
        let recovered = try! engine.decode(codeword: pairs, dataLen: k)

        expect(recovered.count == k, "BN254: recovered length = k")
        var allMatch = true
        for i in 0..<k {
            if recovered[i] != data[i] { allMatch = false; break }
        }
        expect(allMatch, "BN254: encode then decode recovers original data")
        print("  BN254 encode/decode round-trip: OK")
    }

    // ------------------------------------------------------------------
    // 2. BabyBear: Encode then decode recovers original data
    // ------------------------------------------------------------------
    do {
        let engine = try! ReedSolomonBbEngine()
        let k = 8
        var data = [Bb]()
        for i in 0..<k {
            data.append(Bb(v: UInt32(i + 1)))
        }

        let codeword = try! engine.encode(data: data, codeRate: 2)
        let n = codeword.count
        expect(n == 16, "BabyBear: codeword length = 16")

        let pairs: [(Int, Bb)] = (0..<k).map { i in (i, codeword[i]) }
        let recovered = try! engine.decode(codeword: pairs, dataLen: k)

        expect(recovered.count == k, "BabyBear: recovered length = k")
        var allMatch = true
        for i in 0..<k {
            if recovered[i] != data[i] { allMatch = false; break }
        }
        expect(allMatch, "BabyBear: encode then decode recovers original data")
        print("  BabyBear encode/decode round-trip: OK")
    }

    // ------------------------------------------------------------------
    // 3. BN254: Decode from random k-of-n subsets (multiple patterns)
    // ------------------------------------------------------------------
    do {
        let engine = try! ReedSolomonEngine()
        let k = 8
        var data = [Fr]()
        for i in 0..<k {
            data.append(frFromInt(UInt64(i * 7 + 3)))
        }

        let codeword = try! engine.encode(data: data, codeRate: 4)
        let n = codeword.count  // 32

        var rng: UInt64 = 12345
        var patternsOk = 0
        let numPatterns = 5

        for pattern in 0..<numPatterns {
            // Generate a random k-subset of [0, n)
            var indices = Set<Int>()
            while indices.count < k {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                let idx = Int((rng >> 33) % UInt64(n))
                indices.insert(idx)
            }

            let pairs: [(Int, Fr)] = indices.sorted().map { i in (i, codeword[i]) }
            let recovered = try! engine.decode(codeword: pairs, dataLen: k)

            var ok = true
            for i in 0..<k {
                if recovered[i] != data[i] { ok = false; break }
            }
            if ok { patternsOk += 1 }
        }

        expect(patternsOk == numPatterns, "BN254: all \(numPatterns) random erasure patterns decoded correctly")
        print("  BN254 random erasure patterns (\(numPatterns)x): OK")
    }

    // ------------------------------------------------------------------
    // 4. BabyBear: Decode from random k-of-n subsets
    // ------------------------------------------------------------------
    do {
        let engine = try! ReedSolomonBbEngine()
        let k = 8
        var data = [Bb]()
        for i in 0..<k {
            data.append(Bb(v: UInt32(i * 13 + 5)))
        }

        let codeword = try! engine.encode(data: data, codeRate: 4)
        let n = codeword.count

        var rng: UInt64 = 67890
        var patternsOk = 0
        let numPatterns = 5

        for _ in 0..<numPatterns {
            var indices = Set<Int>()
            while indices.count < k {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                let idx = Int((rng >> 33) % UInt64(n))
                indices.insert(idx)
            }

            let pairs: [(Int, Bb)] = indices.sorted().map { i in (i, codeword[i]) }
            let recovered = try! engine.decode(codeword: pairs, dataLen: k)

            var ok = true
            for i in 0..<k {
                if recovered[i] != data[i] { ok = false; break }
            }
            if ok { patternsOk += 1 }
        }

        expect(patternsOk == numPatterns, "BabyBear: all \(numPatterns) random erasure patterns decoded correctly")
        print("  BabyBear random erasure patterns (\(numPatterns)x): OK")
    }

    // ------------------------------------------------------------------
    // 5. BN254: Verify valid and invalid codewords
    // ------------------------------------------------------------------
    do {
        let engine = try! ReedSolomonEngine()
        let k = 4
        var data = [Fr]()
        for i in 0..<k {
            data.append(frFromInt(UInt64(i + 10)))
        }

        let codeword = try! engine.encode(data: data, codeRate: 2)
        let n = codeword.count

        // Valid codeword should verify
        let valid = try! engine.verify(codeword: codeword, dataLen: k)
        expect(valid, "BN254: valid codeword passes verification")

        // Corrupt one element
        var corrupted = codeword
        corrupted[0] = frAdd(corrupted[0], Fr.one)
        let invalid = try! engine.verify(codeword: corrupted, dataLen: k)
        expect(!invalid, "BN254: corrupted codeword fails verification")

        print("  BN254 verify valid/invalid: OK")
    }

    // ------------------------------------------------------------------
    // 6. BabyBear: Verify valid and invalid codewords
    // ------------------------------------------------------------------
    do {
        let engine = try! ReedSolomonBbEngine()
        let k = 4
        var data = [Bb]()
        for i in 0..<k {
            data.append(Bb(v: UInt32(i + 10)))
        }

        let codeword = try! engine.encode(data: data, codeRate: 2)

        let valid = try! engine.verify(codeword: codeword, dataLen: k)
        expect(valid, "BabyBear: valid codeword passes verification")

        var corrupted = codeword
        corrupted[0] = bbAdd(corrupted[0], Bb.one)
        let invalid = try! engine.verify(codeword: corrupted, dataLen: k)
        expect(!invalid, "BabyBear: corrupted codeword fails verification")

        print("  BabyBear verify valid/invalid: OK")
    }

    // ------------------------------------------------------------------
    // 7. Multi-block erasure coding round-trip (BN254)
    // ------------------------------------------------------------------
    do {
        let config = ErasureConfig(blockSize: 4, blowupFactor: 2)
        let encoder = try! ErasureEncoder(config: config)
        let decoder = try! ErasureDecoder(config: config)

        // 10 data elements across 3 blocks (last block padded)
        let dataLen = 10
        var data = [Fr]()
        for i in 0..<dataLen {
            data.append(frFromInt(UInt64(i * 3 + 1)))
        }

        let (shards, merkleRoot) = try! encoder.encode(data: data)
        let totalShards = shards.count
        expect(totalShards == config.totalShards, "Multi-block: correct shard count")
        expect(merkleRoot.count == 32, "Multi-block: Merkle root is 32 bytes")

        // Use only the minimum number of shards (select first minShards)
        let subset = Array(shards.prefix(config.minShards))
        let recovered = try! decoder.decode(shards: subset, originalDataLen: dataLen)

        expect(recovered.count == dataLen, "Multi-block: recovered correct length")
        var allMatch = true
        for i in 0..<dataLen {
            if recovered[i] != data[i] { allMatch = false; break }
        }
        expect(allMatch, "Multi-block: encode/decode round-trip recovers original data")
        print("  Multi-block erasure coding round-trip: OK")
    }

    // ------------------------------------------------------------------
    // 8. Multi-block erasure coding round-trip (BabyBear)
    // ------------------------------------------------------------------
    do {
        let config = ErasureConfig(blockSize: 4, blowupFactor: 2)
        let encoder = try! ErasureEncoderBb(config: config)
        let decoder = try! ErasureDecoderBb(config: config)

        let dataLen = 10
        var data = [Bb]()
        for i in 0..<dataLen {
            data.append(Bb(v: UInt32(i * 5 + 2)))
        }

        let (shards, merkleRoot) = try! encoder.encode(data: data)
        expect(merkleRoot.count == 32, "BabyBear multi-block: Merkle root is 32 bytes")

        let subset = Array(shards.prefix(config.minShards))
        let recovered = try! decoder.decode(shards: subset, originalDataLen: dataLen)

        expect(recovered.count == dataLen, "BabyBear multi-block: recovered correct length")
        var allMatch = true
        for i in 0..<dataLen {
            if recovered[i] != data[i] { allMatch = false; break }
        }
        expect(allMatch, "BabyBear multi-block: round-trip recovers original data")
        print("  BabyBear multi-block erasure coding: OK")
    }

    // ------------------------------------------------------------------
    // 9. NTT-based encode matches naive polynomial evaluation (BN254)
    // ------------------------------------------------------------------
    do {
        let engine = try! ReedSolomonEngine()
        let k = 4
        var data = [Fr]()
        for i in 0..<k {
            data.append(frFromInt(UInt64(i + 1)))
        }

        // NTT encode (GPU/CPU via engine)
        let nttResult = try! engine.encode(data: data, codeRate: 2)
        let n = nttResult.count
        let logN = logBase2(n)
        let omega = frRootOfUnity(logN: logN)

        // Naive polynomial evaluation at each root of unity
        var naiveResult = [Fr](repeating: .zero, count: n)
        var padded = [Fr](repeating: .zero, count: n)
        for i in 0..<k { padded[i] = data[i] }

        var omegaI = Fr.one
        for i in 0..<n {
            var acc = Fr.zero
            for j in stride(from: n - 1, through: 0, by: -1) {
                acc = frAdd(frMul(acc, omegaI), padded[j])
            }
            naiveResult[i] = acc
            omegaI = frMul(omegaI, omega)
        }

        var match = true
        for i in 0..<n {
            if nttResult[i] != naiveResult[i] { match = false; break }
        }
        expect(match, "BN254: NTT encode matches naive polynomial evaluation")
        print("  NTT vs naive polynomial evaluation: OK")
    }

    // ------------------------------------------------------------------
    // 10. NTT-based encode matches naive evaluation (BabyBear)
    // ------------------------------------------------------------------
    do {
        let engine = try! ReedSolomonBbEngine()
        let k = 4
        var data = [Bb]()
        for i in 0..<k {
            data.append(Bb(v: UInt32(i + 1)))
        }

        let nttResult = try! engine.encode(data: data, codeRate: 2)
        let n = nttResult.count
        let logN = logBase2(n)
        let omega = bbRootOfUnity(logN: logN)

        var padded = [Bb](repeating: .zero, count: n)
        for i in 0..<k { padded[i] = data[i] }

        var naiveResult = [Bb](repeating: .zero, count: n)
        var omegaI = Bb.one
        for i in 0..<n {
            var acc = Bb.zero
            for j in stride(from: n - 1, through: 0, by: -1) {
                acc = bbAdd(bbMul(acc, omegaI), padded[j])
            }
            naiveResult[i] = acc
            omegaI = bbMul(omegaI, omega)
        }

        var match = true
        for i in 0..<n {
            if nttResult[i] != naiveResult[i] { match = false; break }
        }
        expect(match, "BabyBear: NTT encode matches naive polynomial evaluation")
        print("  BabyBear NTT vs naive evaluation: OK")
    }

    // ------------------------------------------------------------------
    // 11. Insufficient shards error
    // ------------------------------------------------------------------
    do {
        let engine = try! ReedSolomonEngine()
        let k = 8
        var data = [Fr]()
        for i in 0..<k { data.append(frFromInt(UInt64(i))) }

        let codeword = try! engine.encode(data: data, codeRate: 2)

        // Only provide k-1 shards
        let pairs: [(Int, Fr)] = (0..<(k - 1)).map { i in (i, codeword[i]) }
        var threw = false
        do {
            _ = try engine.decode(codeword: pairs, dataLen: k)
        } catch RSEngineError.insufficientSymbols {
            threw = true
        } catch {}
        expect(threw, "BN254: decode with insufficient shards throws error")
        print("  Insufficient shards error: OK")
    }

    // ------------------------------------------------------------------
    // 12. Code rate validation
    // ------------------------------------------------------------------
    do {
        let engine = try! ReedSolomonEngine()
        let data = [frFromInt(1), frFromInt(2)]

        // Invalid code rate (not power of 2)
        var threw = false
        do {
            _ = try engine.encode(data: data, codeRate: 3)
        } catch RSEngineError.invalidCodeRate {
            threw = true
        } catch {}
        expect(threw, "BN254: invalid code rate (3) throws error")

        // Code rate 1 should fail
        threw = false
        do {
            _ = try engine.encode(data: data, codeRate: 1)
        } catch RSEngineError.invalidCodeRate {
            threw = true
        } catch {}
        expect(threw, "BN254: code rate 1 throws error")

        print("  Code rate validation: OK")
    }

    // ------------------------------------------------------------------
    // 13. 4x and 8x blowup factors
    // ------------------------------------------------------------------
    do {
        let engine = try! ReedSolomonEngine()
        let k = 8
        var data = [Fr]()
        for i in 0..<k { data.append(frFromInt(UInt64(i + 100))) }

        // 4x blowup
        let cw4 = try! engine.encode(data: data, codeRate: 4)
        expect(cw4.count == 32, "BN254: 4x blowup gives 32 shards")
        let valid4 = try! engine.verify(codeword: cw4, dataLen: k)
        expect(valid4, "BN254: 4x codeword verifies")

        // Decode from last k shards
        let pairs4: [(Int, Fr)] = (24..<32).map { i in (i, cw4[i]) }
        let rec4 = try! engine.decode(codeword: pairs4, dataLen: k)
        var ok4 = true
        for i in 0..<k { if rec4[i] != data[i] { ok4 = false; break } }
        expect(ok4, "BN254: 4x decode from tail shards")

        // 8x blowup
        let cw8 = try! engine.encode(data: data, codeRate: 8)
        expect(cw8.count == 64, "BN254: 8x blowup gives 64 shards")
        let valid8 = try! engine.verify(codeword: cw8, dataLen: k)
        expect(valid8, "BN254: 8x codeword verifies")

        print("  4x and 8x blowup factors: OK")
    }

    // ------------------------------------------------------------------
    // 14. Erasure verifier — codeword verification
    // ------------------------------------------------------------------
    do {
        let config = ErasureConfig(blockSize: 4, blowupFactor: 2)
        let verifier = try! ErasureVerifier(config: config)

        let k = 4
        var data = [Fr]()
        for i in 0..<k { data.append(frFromInt(UInt64(i + 1))) }

        let codeword = try! verifier.rsEngine.encode(data: data, codeRate: 2)
        let valid = try! verifier.verifyCodeword(codeword: codeword, dataLen: k)
        expect(valid, "Verifier: valid codeword passes")

        var corrupted = codeword
        corrupted[2] = frAdd(corrupted[2], frFromInt(999))
        let invalid = try! verifier.verifyCodeword(codeword: corrupted, dataLen: k)
        expect(!invalid, "Verifier: corrupted codeword fails")

        print("  ErasureVerifier codeword check: OK")
    }

    // ------------------------------------------------------------------
    // 15. Performance benchmark: 2^14 data elements at 4x rate (BabyBear, GPU)
    // ------------------------------------------------------------------
    do {
        let engine = try! ReedSolomonBbEngine()
        let k = 1 << 14  // 16384

        var data = [Bb]()
        data.reserveCapacity(k)
        var rng: UInt32 = 42
        for _ in 0..<k {
            rng = rng &* 1664525 &+ 1013904223
            data.append(Bb(v: rng % Bb.P))
        }

        // Warm up
        _ = try! engine.encode(data: data, codeRate: 4)

        let iterations = 3
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = try! engine.encode(data: data, codeRate: 4)
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - t0) / Double(iterations)
        let throughput = Double(k) / elapsed / 1e6

        print(String(format: "  Benchmark: 2^14 BabyBear @ 4x rate: %.2f ms (%.1f M elem/s)", elapsed * 1000, throughput))
        expect(true, "Benchmark completed")
    }

    // ------------------------------------------------------------------
    // 16. Performance benchmark: 2^14 data elements at 4x rate (BN254 Fr, GPU)
    // ------------------------------------------------------------------
    do {
        let engine = try! ReedSolomonEngine()
        let k = 1 << 14

        var data = [Fr]()
        data.reserveCapacity(k)
        for i in 0..<k {
            data.append(frFromInt(UInt64(i + 1)))
        }

        // Warm up
        _ = try! engine.encode(data: data, codeRate: 4)

        let iterations = 3
        let t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            _ = try! engine.encode(data: data, codeRate: 4)
        }
        let elapsed = (CFAbsoluteTimeGetCurrent() - t0) / Double(iterations)
        let throughput = Double(k) / elapsed / 1e6

        print(String(format: "  Benchmark: 2^14 BN254 Fr @ 4x rate: %.2f ms (%.1f M elem/s)", elapsed * 1000, throughput))
        expect(true, "Benchmark completed")
    }
}
