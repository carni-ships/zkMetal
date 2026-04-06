// GPU FRI Fold Engine tests
// Validates fold correctness against CPU reference, domain inverse precomputation,
// multi-round folding, and GPU-resident fold-to-final.
// Run: swift build && .build/debug/zkMetalTests

import zkMetal
import Foundation

// MARK: - Test RNG

private struct FoldRNG {
    var state: UInt64

    mutating func next32() -> UInt32 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return UInt32(state >> 33)
    }

    mutating func nextFr() -> Fr {
        let raw = Fr(v: (next32() & 0x0FFFFFFF, next32(), next32(), next32(),
                         next32(), next32(), next32(), next32() & 0x0FFFFFFF))
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}

// MARK: - CPU Reference

/// CPU reference FRI fold using explicit domain inverses.
/// result[i] = (evals[i] + evals[i + n/2]) + challenge * (evals[i] - evals[i + n/2]) * domainInv[i]
private func cpuFoldReference(evals: [Fr], domainInv: [Fr], challenge: Fr) -> [Fr] {
    let n = evals.count
    let half = n / 2
    var folded = [Fr](repeating: Fr.zero, count: half)
    for i in 0..<half {
        let a = evals[i]
        let b = evals[i + half]
        let sum = frAdd(a, b)
        let diff = frSub(a, b)
        let term = frMul(challenge, frMul(diff, domainInv[i]))
        folded[i] = frAdd(sum, term)
    }
    return folded
}

/// Build multiplicative domain: omega^0, omega^1, ..., omega^{n-1}
private func buildDomain(logN: Int) -> [Fr] {
    let n = 1 << logN
    let invTwiddles = precomputeInverseTwiddles(logN: logN)
    // omega = 1 / invTwiddle[1]
    let omega = frInverse(invTwiddles[1])
    var domain = [Fr](repeating: Fr.one, count: n)
    var w = Fr.one
    for i in 0..<n {
        domain[i] = w
        w = frMul(w, omega)
    }
    return domain
}

// MARK: - Tests

private func testDomainInversePrecomputation() {
    suite("FRI Fold: Domain inverse precomputation")
    do {
        let engine = try GPUFRIFoldEngine()
        var rng = FoldRNG(state: 42)
        let count = 1024

        // Build random domain elements (non-zero)
        var domain = [Fr](repeating: Fr.one, count: count)
        for i in 0..<count {
            domain[i] = rng.nextFr()
            // Ensure non-zero
            if domain[i].isZero { domain[i] = Fr.one }
        }

        // GPU precompute
        let gpuInvBuf = engine.precomputeDomainInverses(domain: domain)
        let gpuPtr = gpuInvBuf.contents().bindMemory(to: Fr.self, capacity: count)

        // CPU reference
        var allMatch = true
        for i in 0..<count {
            let cpuInv = frInverse(domain[i])
            if gpuPtr[i] != cpuInv {
                allMatch = false
                print("  Mismatch at index \(i)")
                break
            }
        }
        expect(allMatch, "GPU domain inverses match CPU reference (n=\(count))")
        print("  Domain inverse precomputation: \(allMatch ? "PASS" : "FAIL") (n=\(count))")
    } catch {
        print("  [FAIL] Domain inverse precomputation: \(error)")
        expect(false, "Domain inverse threw: \(error)")
    }
}

private func testSingleFoldRound() {
    suite("FRI Fold: Single fold round")
    do {
        let engine = try GPUFRIFoldEngine()
        var rng = FoldRNG(state: 77)
        let logN = 12
        let n = 1 << logN
        let half = n / 2

        // Generate random evaluations
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }
        let challenge = rng.nextFr()

        // Build domain and domain inverses
        let domain = buildDomain(logN: logN)
        let halfDomain = Array(domain[0..<half])
        var domainInv = [Fr](repeating: Fr.zero, count: half)
        for i in 0..<half {
            domainInv[i] = frInverse(halfDomain[i])
        }

        // GPU fold
        let evalsBuf = engine.device.makeBuffer(
            bytes: evals, length: n * MemoryLayout<Fr>.stride,
            options: .storageModeShared)!
        let domainInvBuf = engine.device.makeBuffer(
            bytes: domainInv, length: half * MemoryLayout<Fr>.stride,
            options: .storageModeShared)!

        let gpuResult = try engine.fold(evals: evalsBuf, domainInv: domainInvBuf,
                                         challenge: challenge, n: n)

        // CPU reference
        let cpuResult = cpuFoldReference(evals: evals, domainInv: domainInv, challenge: challenge)

        // Compare
        let ptr = gpuResult.contents().bindMemory(to: Fr.self, capacity: half)
        var allMatch = true
        for i in 0..<half {
            if ptr[i] != cpuResult[i] {
                allMatch = false
                print("  Mismatch at index \(i)")
                break
            }
        }
        expect(allMatch, "Single fold matches CPU reference (n=\(n))")
        print("  Single fold round: \(allMatch ? "PASS" : "FAIL") (n=\(n))")
    } catch {
        print("  [FAIL] Single fold round: \(error)")
        expect(false, "Single fold threw: \(error)")
    }
}

private func testFoldWithLogN() {
    suite("FRI Fold: Fold using logN (auto domain)")
    do {
        let engine = try GPUFRIFoldEngine()
        var rng = FoldRNG(state: 101)
        let logN = 14
        let n = 1 << logN
        let half = n / 2

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }
        let challenge = rng.nextFr()

        let evalsBuf = engine.device.makeBuffer(
            bytes: evals, length: n * MemoryLayout<Fr>.stride,
            options: .storageModeShared)!

        // Use convenience method that auto-generates domain
        let gpuResult = try engine.fold(evals: evalsBuf, logN: logN, challenge: challenge)

        // CPU reference with explicit domain
        let domain = buildDomain(logN: logN)
        var domainInv = [Fr](repeating: Fr.zero, count: half)
        for i in 0..<half {
            domainInv[i] = frInverse(domain[i])
        }
        let cpuResult = cpuFoldReference(evals: evals, domainInv: domainInv, challenge: challenge)

        let ptr = gpuResult.contents().bindMemory(to: Fr.self, capacity: half)
        var allMatch = true
        for i in 0..<half {
            if ptr[i] != cpuResult[i] {
                allMatch = false
                break
            }
        }
        expect(allMatch, "Auto-domain fold matches CPU (logN=\(logN))")
        print("  Auto-domain fold: \(allMatch ? "PASS" : "FAIL") (logN=\(logN))")
    } catch {
        print("  [FAIL] Auto-domain fold: \(error)")
        expect(false, "Auto-domain fold threw: \(error)")
    }
}

private func testFoldAllRounds() {
    suite("FRI Fold: Full multi-round foldAllRounds")
    do {
        let engine = try GPUFRIFoldEngine()
        var rng = FoldRNG(state: 200)
        let logN = 10
        let n = 1 << logN
        let numRounds = 4

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        var challenges = [Fr]()
        for _ in 0..<numRounds { challenges.append(rng.nextFr()) }

        // GPU: foldAllRounds with explicit domain
        let domain = buildDomain(logN: logN)
        let layers = try engine.foldAllRounds(evals: evals, domain: domain,
                                                challenges: challenges)

        // Verify layer count
        expect(layers.count == numRounds + 1, "layers.count == \(numRounds + 1)")

        // Verify initial layer is input
        expect(layers[0] == evals, "Layer 0 is input evals")

        // Verify each layer size halves
        for k in 1...numRounds {
            let expectedSize = n >> k
            expect(layers[k].count == expectedSize,
                   "Layer \(k) size is \(expectedSize)")
        }

        // CPU reference: fold sequentially
        var cpuEvals = evals
        var cpuDomain = domain
        for round in 0..<numRounds {
            let currentN = cpuEvals.count
            let half = currentN / 2
            var domainInv = [Fr](repeating: Fr.zero, count: half)
            for i in 0..<half {
                domainInv[i] = frInverse(cpuDomain[i])
            }
            cpuEvals = cpuFoldReference(evals: cpuEvals, domainInv: domainInv,
                                         challenge: challenges[round])
            // Square the domain for next round
            if round < numRounds - 1 {
                var nextDomain = [Fr](repeating: Fr.zero, count: half)
                for i in 0..<half {
                    nextDomain[i] = frMul(cpuDomain[i], cpuDomain[i])
                }
                cpuDomain = nextDomain
            }
        }

        // Compare final layer
        let finalLayer = layers[numRounds]
        var allMatch = true
        for i in 0..<finalLayer.count {
            if finalLayer[i] != cpuEvals[i] {
                allMatch = false
                print("  Mismatch at final layer index \(i)")
                break
            }
        }
        expect(allMatch, "Final layer matches CPU after \(numRounds) rounds")
        print("  foldAllRounds: \(allMatch ? "PASS" : "FAIL") (\(numRounds) rounds, logN=\(logN))")
    } catch {
        print("  [FAIL] foldAllRounds: \(error)")
        expect(false, "foldAllRounds threw: \(error)")
    }
}

private func testFoldAllRoundsLogN() {
    suite("FRI Fold: foldAllRounds with logN convenience")
    do {
        let engine = try GPUFRIFoldEngine()
        var rng = FoldRNG(state: 333)
        let logN = 12
        let n = 1 << logN
        let numRounds = 5

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        var challenges = [Fr]()
        for _ in 0..<numRounds { challenges.append(rng.nextFr()) }

        let layers = try engine.foldAllRounds(evals: evals, logN: logN,
                                                challenges: challenges)

        // Verify structure
        expect(layers.count == numRounds + 1, "Correct layer count")
        let finalSize = n >> numRounds
        expect(layers[numRounds].count == finalSize, "Final layer size correct")

        // Verify non-zero results
        let hasNonZero = layers[numRounds].contains { !$0.isZero }
        expect(hasNonZero, "Final layer has non-zero elements")
        print("  foldAllRounds(logN): PASS (logN=\(logN), \(numRounds) rounds, final=\(finalSize))")
    } catch {
        print("  [FAIL] foldAllRounds(logN): \(error)")
        expect(false, "foldAllRounds(logN) threw: \(error)")
    }
}

private func testFoldToFinal() {
    suite("FRI Fold: foldToFinal (GPU-resident)")
    do {
        let engine = try GPUFRIFoldEngine()
        var rng = FoldRNG(state: 444)
        let logN = 14
        let n = 1 << logN
        let numRounds = logN  // fold all the way to 1 element

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        var challenges = [Fr]()
        for _ in 0..<numRounds { challenges.append(rng.nextFr()) }

        let evalsBuf = engine.device.makeBuffer(
            bytes: evals, length: n * MemoryLayout<Fr>.stride,
            options: .storageModeShared)!

        let resultBuf = try engine.foldToFinal(evals: evalsBuf, logN: logN,
                                                 challenges: challenges)

        // Read single result
        let ptr = resultBuf.contents().bindMemory(to: Fr.self, capacity: 1)
        let result = ptr[0]
        expect(!result.isZero, "Fold to final produces non-zero constant")

        // Cross-check with foldAllRounds
        let layers = try engine.foldAllRounds(evals: evals, logN: logN,
                                                challenges: challenges)
        expect(layers.last!.count == 1, "foldAllRounds produces 1 element")
        let match = result == layers.last![0]
        expect(match, "foldToFinal matches foldAllRounds result")
        print("  foldToFinal: \(match ? "PASS" : "FAIL") (logN=\(logN), folded to 1 element)")
    } catch {
        print("  [FAIL] foldToFinal: \(error)")
        expect(false, "foldToFinal threw: \(error)")
    }
}

private func testCPUFallback() {
    suite("FRI Fold: CPU fallback for small domains")
    do {
        let engine = try GPUFRIFoldEngine()
        var rng = FoldRNG(state: 555)
        let logN = 8  // 256 elements, below threshold
        let n = 1 << logN
        let half = n / 2

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }
        let challenge = rng.nextFr()

        let evalsBuf = engine.device.makeBuffer(
            bytes: evals, length: n * MemoryLayout<Fr>.stride,
            options: .storageModeShared)!

        // Should use CPU fallback
        let gpuResult = try engine.fold(evals: evalsBuf, logN: logN, challenge: challenge)

        // CPU reference with explicit domain
        let domain = buildDomain(logN: logN)
        var domainInv = [Fr](repeating: Fr.zero, count: half)
        for i in 0..<half {
            domainInv[i] = frInverse(domain[i])
        }
        let cpuResult = cpuFoldReference(evals: evals, domainInv: domainInv, challenge: challenge)

        let ptr = gpuResult.contents().bindMemory(to: Fr.self, capacity: half)
        var allMatch = true
        for i in 0..<half {
            if ptr[i] != cpuResult[i] {
                allMatch = false
                break
            }
        }
        expect(allMatch, "CPU fallback matches reference (n=\(n))")
        print("  CPU fallback: \(allMatch ? "PASS" : "FAIL") (n=\(n))")
    } catch {
        print("  [FAIL] CPU fallback: \(error)")
        expect(false, "CPU fallback threw: \(error)")
    }
}

private func testPerformance() {
    suite("FRI Fold: Performance 2^20 fold-to-final")
    do {
        let engine = try GPUFRIFoldEngine()
        var rng = FoldRNG(state: 999)
        let logN = 20
        let n = 1 << logN
        let numRounds = 10

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        var challenges = [Fr]()
        for _ in 0..<numRounds { challenges.append(rng.nextFr()) }

        let evalsBuf = engine.device.makeBuffer(
            bytes: evals, length: n * MemoryLayout<Fr>.stride,
            options: .storageModeShared)!

        // Warmup
        _ = try engine.foldToFinal(evals: evalsBuf, logN: logN, challenges: challenges)

        // Re-upload
        let evalsBuf2 = engine.device.makeBuffer(
            bytes: evals, length: n * MemoryLayout<Fr>.stride,
            options: .storageModeShared)!

        let start = DispatchTime.now()
        let result = try engine.foldToFinal(evals: evalsBuf2, logN: logN,
                                              challenges: challenges)
        let end = DispatchTime.now()
        let ms = Double(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000.0

        let finalSize = 1 << (logN - numRounds)
        let ptr = result.contents().bindMemory(to: Fr.self, capacity: finalSize)
        let nonZero = !ptr[0].isZero
        expect(nonZero, "Performance fold produces non-zero result")
        print("  Performance: 2^20 x \(numRounds) rounds in \(String(format: "%.2f", ms)) ms")
        print("  Final size: \(finalSize) elements")
    } catch {
        print("  [FAIL] Performance: \(error)")
        expect(false, "Performance fold threw: \(error)")
    }
}

// MARK: - Public entry point

public func runGPUFRIFoldTests() {
    testDomainInversePrecomputation()
    testSingleFoldRound()
    testFoldWithLogN()
    testFoldAllRounds()
    testFoldAllRoundsLogN()
    testFoldToFinal()
    testCPUFallback()
    testPerformance()
}
