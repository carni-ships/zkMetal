// Reed-Solomon Erasure Coding Benchmark
import zkMetal
import Foundation

public func runErasureBench() {
    print("=== Reed-Solomon Erasure Coding Benchmark ===")

    // --- GF(2^16) Correctness ---
    print("\n--- GF(2^16) Field Correctness ---")
    let a = GF16(value: 1234)
    let b = GF16(value: 5678)
    let prod = gf16Mul(a, b)
    let inv = gf16Inverse(b)
    let check = gf16Mul(prod, inv)
    print("  a=\(a.value) b=\(b.value) a*b=\(prod.value) a*b/b=\(check.value) == a? \(check.value == a.value ? "PASS" : "FAIL")")

    // Associativity: (a*b)*c == a*(b*c)
    let c = GF16(value: 9999)
    let lhs = gf16Mul(gf16Mul(a, b), c)
    let rhs = gf16Mul(a, gf16Mul(b, c))
    print("  Associativity: \(lhs.value == rhs.value ? "PASS" : "FAIL")")

    // Inverse: a * a^-1 == 1
    let aInv = gf16Inverse(a)
    let one = gf16Mul(a, aInv)
    print("  a * a^-1 == 1: \(one.value == 1 ? "PASS" : "FAIL")")

    // --- NTT-Based RS (BabyBear) ---
    print("\n--- NTT-Based RS (BabyBear) ---")
    do {
        let engine = try ReedSolomonNTTEngine()

        // Correctness: encode then decode
        let k = 64
        var data = [Bb](repeating: .zero, count: k)
        var rng: UInt64 = 0xCAFE_BABE
        for i in 0..<k {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            data[i] = Bb(v: UInt32(rng >> 33) % Bb.P)
        }

        let encoded = try engine.encode(data: data, expansionFactor: 2)
        let n = encoded.count
        print("  Encode: k=\(k) -> n=\(n) shards")

        // Decode from first k shards
        let shards = (0..<k).map { (index: $0, value: encoded[$0]) }
        let decoded = try engine.decode(shards: shards, originalK: k, totalN: n)
        var correct = true
        for i in 0..<k {
            if decoded[i].v != data[i].v { correct = false; break }
        }
        print("  Decode (first k): \(correct ? "PASS" : "FAIL")")

        // Decode from last k shards (simulating first k lost)
        let lastShards = (0..<k).map { (index: n - k + $0, value: encoded[n - k + $0]) }
        let decoded2 = try engine.decode(shards: lastShards, originalK: k, totalN: n)
        var correct2 = true
        for i in 0..<k {
            if decoded2[i].v != data[i].v { correct2 = false; break }
        }
        print("  Decode (last k): \(correct2 ? "PASS" : "FAIL")")

        // Benchmark encoding
        print("\n--- NTT RS Encode Benchmark (BabyBear) ---")
        for logK in [8, 10, 12, 14] {
            let kSize = 1 << logK
            var testData = [Bb](repeating: .zero, count: kSize)
            for i in 0..<kSize {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                testData[i] = Bb(v: UInt32(rng >> 33) % Bb.P)
            }

            for expansion in [2, 4] {
                // Warmup
                let _ = try engine.encode(data: testData, expansionFactor: expansion)

                let runs = 5
                var times = [Double]()
                for _ in 0..<runs {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let _ = try engine.encode(data: testData, expansionFactor: expansion)
                    times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                }
                times.sort()
                let median = times[runs / 2]
                var totalN = kSize * expansion
                // Round up to next power of 2
                totalN = 1 << (Int.bitWidth - (totalN - 1).leadingZeroBitCount)
                let bytes = Double(kSize * 4)  // 4 bytes per BabyBear
                let mbps = bytes / (median / 1000) / 1_000_000
                print(String(format: "  k=2^%-2d n=%6d (%dx): %7.2f ms  (%5.1f MB/s)",
                             logK, totalN, expansion, median, mbps))
            }
        }
    } catch {
        print("  NTT RS ERROR: \(error)")
    }

    // --- GF(2^16) GPU RS ---
    print("\n--- GF(2^16) GPU RS Benchmark ---")
    do {
        let engine = try ReedSolomonGF16Engine()

        // Correctness
        let k = 128
        let parity = 128
        var data = [GF16](repeating: .zero, count: k)
        var rng: UInt64 = 0xDEAD_BEEF
        for i in 0..<k {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            data[i] = GF16(value: UInt16(rng >> 48) | 1)  // nonzero
        }

        let encoded = try engine.encode(data: data, parityCount: parity)
        print("  Encode: k=\(k) parity=\(parity) -> \(encoded.count) shards")

        // Verify first k shards are systematic (same as data)
        var systematic = true
        for i in 0..<k {
            if encoded[i].value != data[i].value { systematic = false; break }
        }
        print("  Systematic: \(systematic ? "PASS" : "FAIL")")

        // Decode from first k shards
        let shards = (0..<k).map { (index: $0, value: encoded[$0]) }
        let decoded = try engine.decode(shards: shards, originalK: k, totalN: k + parity)
        var correct = true
        for i in 0..<k {
            if decoded[i].value != data[i].value { correct = false; break }
        }
        print("  Decode (systematic): \(correct ? "PASS" : "FAIL")")

        // Decode from k shards with some parity mixed in
        var mixedShards = [(index: Int, value: GF16)]()
        // Take every other data shard + fill with parity
        for i in stride(from: 0, to: k, by: 2) {
            mixedShards.append((index: i, value: encoded[i]))
        }
        for i in 0..<(k - mixedShards.count) {
            mixedShards.append((index: k + i, value: encoded[k + i]))
        }
        let decoded2 = try engine.decode(shards: mixedShards, originalK: k, totalN: k + parity)
        var correct2 = true
        for i in 0..<k {
            if decoded2[i].value != data[i].value { correct2 = false; break }
        }
        print("  Decode (mixed erasure): \(correct2 ? "PASS" : "FAIL")")

        // GPU batch multiply correctness
        let aMul = [GF16(value: 100), GF16(value: 200), GF16(value: 300)]
        let bMul = [GF16(value: 400), GF16(value: 500), GF16(value: 600)]
        let gpuMul = try engine.batchMul(aMul, bMul)
        var batchCorrect = true
        for i in 0..<3 {
            let expected = gf16Mul(aMul[i], bMul[i])
            if gpuMul[i].value != expected.value { batchCorrect = false }
        }
        print("  GPU batch mul: \(batchCorrect ? "PASS" : "FAIL")")

        // Benchmark
        print("\n--- GF(2^16) RS Encode Throughput ---")
        for kSize in [256, 512, 1024, 2048, 4096] {
            for parityRatio in [1, 3] {  // 1x = 2x expansion, 3x = 4x expansion
                let parityN = kSize * parityRatio
                var testData = [GF16](repeating: .zero, count: kSize)
                for i in 0..<kSize {
                    rng = rng &* 6364136223846793005 &+ 1442695040888963407
                    testData[i] = GF16(value: UInt16(rng >> 48) | 1)
                }

                // Warmup
                let _ = try engine.encode(data: testData, parityCount: parityN)

                let runs = 5
                var times = [Double]()
                for _ in 0..<runs {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let _ = try engine.encode(data: testData, parityCount: parityN)
                    times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                }
                times.sort()
                let median = times[runs / 2]
                let bytes = Double(kSize * 2)  // 2 bytes per GF16
                let mbps = bytes / (median / 1000) / 1_000_000
                let expansion = 1 + parityRatio
                print(String(format: "  k=%4d parity=%5d (%dx): %7.2f ms  (%5.1f MB/s)",
                             kSize, parityN, expansion, median, mbps))
            }
        }

        // Decode benchmark
        print("\n--- GF(2^16) RS Decode Throughput ---")
        for kSize in [256, 1024, 4096] {
            let parityN = kSize
            var testData = [GF16](repeating: .zero, count: kSize)
            for i in 0..<kSize {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                testData[i] = GF16(value: UInt16(rng >> 48) | 1)
            }
            let enc = try engine.encode(data: testData, parityCount: parityN)

            // Use random subset of k shards
            var indices = Array(0..<(kSize + parityN))
            // Fisher-Yates shuffle first k
            for i in 0..<kSize {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                let j = i + Int(rng >> 33) % (indices.count - i)
                indices.swapAt(i, j)
            }
            let decodeShards = (0..<kSize).map { (index: indices[$0], value: enc[indices[$0]]) }

            // Warmup
            let _ = try engine.decode(shards: decodeShards, originalK: kSize, totalN: kSize + parityN)

            let runs = 3
            var times = [Double]()
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.decode(shards: decodeShards, originalK: kSize, totalN: kSize + parityN)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[runs / 2]
            print(String(format: "  k=%4d decode: %7.2f ms", kSize, median))
        }
    } catch {
        print("  GF16 RS ERROR: \(error)")
    }

    // --- Data Availability Sampling ---
    print("\n--- Data Availability Sampling Demo ---")
    do {
        let sampler = try DataAvailabilitySampler(expansionFactor: 2)

        // Create a test blob (simulated blockchain data)
        let blobSize = 4096  // 4KB blob
        var blob = [UInt8](repeating: 0, count: blobSize)
        var rng: UInt64 = 0xDA7A_B10B
        for i in 0..<blobSize {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            blob[i] = UInt8(rng >> 56)
        }

        let result = try sampler.simulate(data: blob, sampleRatio: 0.75)
        print("  Blob size: \(result.originalSize) bytes")
        print("  Encoded shards: \(result.encodedShards)")
        print("  Sampled shards: \(result.sampledShards)")
        print("  Samples verified: \(result.verified ? "PASS" : "FAIL")")
        print("  Reconstruction: \(result.reconstructed ? "PASS" : "FAIL")")

        // Benchmark DAS encode
        print("\n--- DAS Encode Benchmark ---")
        for blobKB in [1, 4, 16, 64, 128] {
            let size = blobKB * 1024
            var testBlob = [UInt8](repeating: 0, count: size)
            for i in 0..<size {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                testBlob[i] = UInt8(rng >> 56)
            }

            // Warmup
            let _ = try sampler.encodeBlob(data: testBlob)

            let runs = 5
            var times = [Double]()
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try sampler.encodeBlob(data: testBlob)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[runs / 2]
            let mbps = Double(size) / (median / 1000) / 1_000_000
            print(String(format: "  %4d KB blob: %7.2f ms  (%5.1f MB/s)", blobKB, median, mbps))
        }
    } catch {
        print("  DAS ERROR: \(error)")
    }

    // --- CPU Baseline ---
    if !skipCPU {
        print("\n--- CPU RS Encode Baseline (GF(2^16)) ---")
        for kSize in [256, 1024, 4096] {
            let parityN = kSize
            var rng: UInt64 = 0xC0DE_BA5E
            var data = [GF16](repeating: .zero, count: kSize)
            for i in 0..<kSize {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                data[i] = GF16(value: UInt16(rng >> 48) | 1)
            }

            let runs = 3
            var times = [Double]()
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = cpuRSEncode(data: data, parityCount: parityN)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[runs / 2]
            let bytes = Double(kSize * 2)
            let mbps = bytes / (median / 1000) / 1_000_000
            print(String(format: "  k=%4d parity=%4d: %7.2f ms  (%5.1f MB/s)", kSize, parityN, median, mbps))
        }
    }
}

/// CPU RS encode for baseline comparison
func cpuRSEncode(data: [GF16], parityCount: Int) -> [GF16] {
    let k = data.count
    let n = k + parityCount
    let evalPoints = gf16EvalPoints(n)

    var result = data
    for i in 0..<parityCount {
        let alpha = evalPoints[k + i]
        var val = GF16.zero
        var alphaPow = GF16.one
        for j in 0..<k {
            val = gf16Add(val, gf16Mul(data[j], alphaPow))
            alphaPow = gf16Mul(alphaPow, alpha)
        }
        result.append(val)
    }
    return result
}

// MARK: - Stubs for missing bench functions (auto-generated engines not yet complete)
// These prevent build errors when main.swift references them.

func _stubMsg(_ name: String) { fputs("[\(name)] not yet available\n", stderr) }
