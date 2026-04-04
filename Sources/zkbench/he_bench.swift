// HE Benchmark — RNS NTT and BFV homomorphic encryption operations

import Foundation
import Metal
import zkMetal

func runHEBench() {
    guard let device = MTLCreateSystemDefaultDevice() else {
        fputs("Error: No Metal GPU available\n", stderr)
        return
    }
    fputs("=== HE / RNS NTT Benchmark — \(device.name) ===\n", stderr)

    // --- Part 1: RNS NTT benchmarks ---
    fputs("\n--- RNS NTT Throughput ---\n", stderr)

    let limbCounts = [1, 3, 5, 10]
    let logNs = [12, 13, 14]  // N=4096, 8192, 16384

    for logN in logNs {
        let n = 1 << logN
        for numLimbs in limbCounts {
            let moduli = heVerifiedPrimes(count: numLimbs, logN: logN)
            if moduli.count < numLimbs {
                fputs("  N=\(n) L=\(numLimbs): not enough primes found, skipping\n", stderr)
                continue
            }

            do {
                let engine = try RNSNTTEngine(logN: logN, moduli: moduli)

                // Create test polynomial
                var poly = RNSPoly(degree: n, moduli: moduli)
                for li in 0..<numLimbs {
                    for j in 0..<n {
                        poly.limbs[li][j] = UInt32.random(in: 0..<moduli[li])
                    }
                }

                // Correctness check: NTT then INTT should round-trip
                var polyNTT = poly
                try engine.forwardNTT(&polyNTT)
                var polyBack = polyNTT
                try engine.inverseNTT(&polyBack)

                var correct = true
                for li in 0..<numLimbs {
                    for j in 0..<min(16, n) {
                        if poly.limbs[li][j] != polyBack.limbs[li][j] {
                            correct = false
                            break
                        }
                    }
                    if !correct { break }
                }

                // Warmup
                var warmup = poly
                try engine.forwardNTT(&warmup)

                // Benchmark forward NTT
                let runs = 10
                var fwdTimes = [Double]()
                for _ in 0..<runs {
                    var p = poly
                    let t0 = CFAbsoluteTimeGetCurrent()
                    try engine.forwardNTT(&p)
                    let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                    fwdTimes.append(elapsed)
                }
                fwdTimes.sort()
                let fwdMedian = fwdTimes[runs / 2]

                // Benchmark inverse NTT
                var invTimes = [Double]()
                for _ in 0..<runs {
                    var p = polyNTT
                    let t0 = CFAbsoluteTimeGetCurrent()
                    try engine.inverseNTT(&p)
                    let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                    invTimes.append(elapsed)
                }
                invTimes.sort()
                let invMedian = invTimes[runs / 2]

                // Total NTT elements per poly
                let totalElements = numLimbs * n
                let throughputFwd = Double(totalElements) / (fwdMedian / 1000) / 1e6  // M elements/s

                let tag = correct ? "OK" : "FAIL"
                fputs("  N=\(n) L=\(numLimbs) [\(tag)]: fwd \(String(format: "%.3f", fwdMedian))ms, inv \(String(format: "%.3f", invMedian))ms, \(String(format: "%.1f", throughputFwd))M elem/s (\(totalElements) elements)\n", stderr)

            } catch {
                fputs("  N=\(n) L=\(numLimbs): ERROR — \(error)\n", stderr)
            }
        }
    }

    // --- Part 2: Pointwise multiply benchmark ---
    fputs("\n--- RNS Pointwise Multiply ---\n", stderr)
    do {
        let logN = 12
        let n = 1 << logN
        let numLimbs = 5
        let moduli = heVerifiedPrimes(count: numLimbs, logN: logN)
        let engine = try RNSNTTEngine(logN: logN, moduli: moduli)

        var a = RNSPoly(degree: n, moduli: moduli)
        var b = RNSPoly(degree: n, moduli: moduli)
        for li in 0..<numLimbs {
            for j in 0..<n {
                a.limbs[li][j] = UInt32.random(in: 0..<moduli[li])
                b.limbs[li][j] = UInt32.random(in: 0..<moduli[li])
            }
        }

        // Warmup
        let _ = try engine.multiply(a, b)

        let runs = 10
        var times = [Double]()
        for _ in 0..<runs {
            let t0 = CFAbsoluteTimeGetCurrent()
            let _ = try engine.multiply(a, b)
            let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            times.append(elapsed)
        }
        times.sort()
        fputs("  N=\(n) L=\(numLimbs): pointwise mul \(String(format: "%.3f", times[runs/2]))ms\n", stderr)
    } catch {
        fputs("  Pointwise mul: ERROR — \(error)\n", stderr)
    }

    // --- Part 3: Batch NTT benchmark ---
    fputs("\n--- Batch NTT (multiple polynomials) ---\n", stderr)
    do {
        let logN = 12
        let n = 1 << logN
        let numLimbs = 5
        let moduli = heVerifiedPrimes(count: numLimbs, logN: logN)
        let engine = try RNSNTTEngine(logN: logN, moduli: moduli)

        for batchSize in [1, 4, 8, 16] {
            var polys = [RNSPoly]()
            for _ in 0..<batchSize {
                var p = RNSPoly(degree: n, moduli: moduli)
                for li in 0..<numLimbs {
                    for j in 0..<n {
                        p.limbs[li][j] = UInt32.random(in: 0..<moduli[li])
                    }
                }
                polys.append(p)
            }

            // Warmup
            var warmup = polys
            try engine.batchForwardNTT(&warmup)

            let runs = 5
            var times = [Double]()
            for _ in 0..<runs {
                var ps = polys
                let t0 = CFAbsoluteTimeGetCurrent()
                try engine.batchForwardNTT(&ps)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                times.append(elapsed)
            }
            times.sort()
            let median = times[runs / 2]
            let perPoly = median / Double(batchSize)
            let totalElements = batchSize * numLimbs * n
            let throughput = Double(totalElements) / (median / 1000) / 1e6

            fputs("  batch=\(batchSize): \(String(format: "%.3f", median))ms total, \(String(format: "%.3f", perPoly))ms/poly, \(String(format: "%.1f", throughput))M elem/s\n", stderr)
        }
    } catch {
        fputs("  Batch NTT: ERROR — \(error)\n", stderr)
    }

    // --- Part 4: BFV HE operations ---
    fputs("\n--- BFV Homomorphic Encryption ---\n", stderr)
    do {
        let logN = 12
        let numLimbs = 3
        let moduli = heVerifiedPrimes(count: numLimbs, logN: logN)
        let params = HEParams(logN: logN, moduli: moduli, plainModulus: 65537)
        let he = try HEEngine(params: params)

        fputs("  params: N=\(1 << logN), L=\(numLimbs), t=\(params.plainModulus)\n", stderr)

        // KeyGen
        let kgT0 = CFAbsoluteTimeGetCurrent()
        let (pk, sk, rlk) = try he.keyGen()
        let kgTime = (CFAbsoluteTimeGetCurrent() - kgT0) * 1000
        fputs("  keygen: \(String(format: "%.1f", kgTime))ms\n", stderr)

        // Encrypt
        let plain1: [Int64] = (0..<(1 << logN)).map { Int64($0 % 100) }
        let plain2: [Int64] = (0..<(1 << logN)).map { Int64(($0 * 3 + 7) % 100) }

        var encTimes = [Double]()
        var ct1 = try he.encrypt(plaintext: plain1, pk: pk)
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            ct1 = try he.encrypt(plaintext: plain1, pk: pk)
            let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            encTimes.append(elapsed)
        }
        encTimes.sort()
        fputs("  encrypt: \(String(format: "%.1f", encTimes[2]))ms\n", stderr)

        let ct2 = try he.encrypt(plaintext: plain2, pk: pk)

        // Decrypt and verify
        let decrypted = try he.decrypt(ciphertext: ct1, sk: sk)
        var decCorrect = true
        for i in 0..<min(10, plain1.count) {
            if decrypted[i] != plain1[i] {
                decCorrect = false
                fputs("  decrypt MISMATCH at [\(i)]: expected \(plain1[i]), got \(decrypted[i])\n", stderr)
                break
            }
        }

        var decTimes = [Double]()
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let _ = try he.decrypt(ciphertext: ct1, sk: sk)
            let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            decTimes.append(elapsed)
        }
        decTimes.sort()
        fputs("  decrypt: \(String(format: "%.1f", decTimes[2]))ms [\(decCorrect ? "OK" : "FAIL")]\n", stderr)

        // Homomorphic add
        var addTimes = [Double]()
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let _ = try he.add(ct1, ct2)
            let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            addTimes.append(elapsed)
        }
        addTimes.sort()

        // Verify add
        let ctAdd = try he.add(ct1, ct2)
        let decAdd = try he.decrypt(ciphertext: ctAdd, sk: sk)
        var addCorrect = true
        for i in 0..<min(10, plain1.count) {
            let expected = (plain1[i] + plain2[i]) % Int64(params.plainModulus)
            if decAdd[i] != expected {
                addCorrect = false
                fputs("  add MISMATCH at [\(i)]: expected \(expected), got \(decAdd[i])\n", stderr)
                break
            }
        }
        fputs("  homo add: \(String(format: "%.1f", addTimes[2]))ms [\(addCorrect ? "OK" : "FAIL")]\n", stderr)

        // Homomorphic multiply
        var mulTimes = [Double]()
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let _ = try he.multiply(ct1, ct2, rlk: rlk)
            let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            mulTimes.append(elapsed)
        }
        mulTimes.sort()
        fputs("  homo mul: \(String(format: "%.1f", mulTimes[2]))ms\n", stderr)

    } catch {
        fputs("  BFV: ERROR — \(error)\n", stderr)
    }

    // --- Part 5: Comparison with ZK NTT throughput ---
    fputs("\n--- Throughput Comparison (NTT elements/s) ---\n", stderr)
    fputs("  (Run 'ntt' and 'circle' benchmarks for ZK field comparison)\n", stderr)
    fputs("  RNS NTT is 4 bytes/element, same as BabyBear and M31\n", stderr)
    fputs("  HE advantage: batch L limbs per ciphertext operation\n", stderr)
}
