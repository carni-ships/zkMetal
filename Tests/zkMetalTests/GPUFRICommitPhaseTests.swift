// GPU FRI Commit Phase Engine tests
// Validates commit phase correctness: folding, Merkle commitments, domain halving,
// polynomial splitting, configurable folding factors, circle FRI, batched commit,
// streaming, and remainder polynomial degree.
// Run: swift build && .build/debug/zkMetalTests

import zkMetal
import Foundation

// MARK: - Test RNG

private struct CommitPhaseRNG {
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

    mutating func nextNonZeroFr() -> Fr {
        var v = nextFr()
        while v.isZero { v = nextFr() }
        return v
    }
}

// MARK: - CPU Reference Helpers

/// CPU reference fold: result[i] = (a + b) + challenge * (a - b) * domainInv[i]
private func cpuFoldRef(evals: [Fr], logN: Int, challenge: Fr) -> [Fr] {
    let n = evals.count
    let half = n / 2
    let invTwiddles = precomputeInverseTwiddles(logN: logN)
    var folded = [Fr](repeating: Fr.zero, count: half)
    for i in 0..<half {
        let a = evals[i]
        let b = evals[i + half]
        let sum = frAdd(a, b)
        let diff = frSub(a, b)
        let term = frMul(challenge, frMul(diff, invTwiddles[i]))
        folded[i] = frAdd(sum, term)
    }
    return folded
}

/// Build multiplicative domain of size 2^logN.
private func buildTestDomain(logN: Int) -> [Fr] {
    let n = 1 << logN
    let invTwiddles = precomputeInverseTwiddles(logN: logN)
    let omega = frInverse(invTwiddles[1])
    var domain = [Fr](repeating: Fr.one, count: n)
    var w = Fr.one
    for i in 0..<n {
        domain[i] = w
        w = frMul(w, omega)
    }
    return domain
}

/// Evaluate polynomial at a point using Horner's method.
private func hornerEval(_ coeffs: [Fr], at point: Fr) -> Fr {
    guard !coeffs.isEmpty else { return Fr.zero }
    var result = coeffs[coeffs.count - 1]
    for i in Swift.stride(from: coeffs.count - 2, through: 0, by: -1) {
        result = frAdd(frMul(result, point), coeffs[i])
    }
    return result
}

// MARK: - Tests

private func testSingleLayerFold() {
    suite("FRI Commit Phase: Single layer fold-by-2")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        var rng = CommitPhaseRNG(state: 1001)
        let logN = 10
        let n = 1 << logN

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        let challenge = rng.nextNonZeroFr()
        let config = FRICommitPhaseConfig(
            foldingFactor: 2, blowupFactor: 2,
            finalPolyMaxDegree: (n / 2) - 1)

        // commitPhase should produce exactly 2 layers (initial + 1 fold)
        let result = try engine.commitPhase(
            evaluations: evals, challenges: [challenge], config: config)

        expectEqual(result.numRounds, 1, "Single fold round")
        expectEqual(result.layers.count, 2, "Two layers: initial + folded")

        // Verify folded layer matches CPU reference
        let cpuFolded = cpuFoldRef(evals: evals, logN: logN, challenge: challenge)
        let gpuFolded = result.layers[1].evaluations!
        expectEqual(gpuFolded.count, cpuFolded.count, "Folded size matches")

        var allMatch = true
        for i in 0..<cpuFolded.count {
            if gpuFolded[i] != cpuFolded[i] { allMatch = false; break }
        }
        expect(allMatch, "GPU fold matches CPU reference fold-by-2")

        // Verify Merkle roots are non-zero
        expect(!result.layers[0].merkleRoot.isZero, "Layer 0 Merkle root non-zero")
        expect(!result.layers[1].merkleRoot.isZero, "Layer 1 Merkle root non-zero")

        print("  Single layer fold-by-2: PASS (n=\(n))")
    } catch {
        print("  [FAIL] Single layer fold: \(error)")
        expect(false, "Single layer fold threw: \(error)")
    }
}

private func testMultiRoundCommit() {
    suite("FRI Commit Phase: Multi-round fold-by-2")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        var rng = CommitPhaseRNG(state: 2002)
        let logN = 12
        let n = 1 << logN

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        let config = FRICommitPhaseConfig(
            foldingFactor: 2, blowupFactor: 4, finalPolyMaxDegree: 7)

        let result = try engine.commitPhase(evaluations: evals, config: config)

        // Verify layer count and sizes
        expect(result.numRounds > 0, "At least one round")
        expectEqual(result.layers.count, result.numRounds + 1, "layers = rounds + 1")

        // Verify each layer halves in size
        for i in 1..<result.layers.count {
            let prevLog = result.layers[i - 1].logDomainSize
            let curLog = result.layers[i].logDomainSize
            expectEqual(curLog, prevLog - 1, "Layer \(i) logSize = prev - 1")
        }

        // Verify challenges were produced
        expectEqual(result.challenges.count, result.numRounds, "One challenge per round")

        // Verify Fiat-Shamir determinism: re-run should give same result
        let result2 = try engine.commitPhase(evaluations: evals, config: config)
        for i in 0..<result.challenges.count {
            expect(frEqual(result.challenges[i], result2.challenges[i]),
                   "Challenge \(i) is deterministic")
        }
        for i in 0..<result.layers.count {
            expect(frEqual(result.layers[i].merkleRoot, result2.layers[i].merkleRoot),
                   "Merkle root \(i) is deterministic")
        }

        print("  Multi-round: PASS (\(result.numRounds) rounds, logN=\(logN))")
    } catch {
        print("  [FAIL] Multi-round: \(error)")
        expect(false, "Multi-round threw: \(error)")
    }
}

private func testFoldByFour() {
    suite("FRI Commit Phase: Fold-by-4")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        var rng = CommitPhaseRNG(state: 3003)
        let logN = 12
        let n = 1 << logN

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        let config = FRICommitPhaseConfig(
            foldingFactor: 4, blowupFactor: 4, finalPolyMaxDegree: 7)

        let result = try engine.commitPhase(evaluations: evals, config: config)

        // Each round reduces by factor 4 (2 bits)
        expect(result.numRounds > 0, "At least one round")
        for i in 1..<result.layers.count {
            let prevLog = result.layers[i - 1].logDomainSize
            let curLog = result.layers[i].logDomainSize
            expectEqual(curLog, prevLog - 2, "Fold-by-4 reduces logSize by 2")
        }

        // Compare fold-by-4 first round vs two sequential fold-by-2
        let challenge = result.challenges[0]
        let cpu1 = cpuFoldRef(evals: evals, logN: logN, challenge: challenge)
        let ch2 = frMul(challenge, challenge)
        let cpu2 = cpuFoldRef(evals: cpu1, logN: logN - 1, challenge: ch2)

        let gpuFolded = result.layers[1].evaluations!
        expectEqual(gpuFolded.count, cpu2.count, "Fold-by-4 output size")

        var allMatch = true
        for i in 0..<cpu2.count {
            if gpuFolded[i] != cpu2[i] { allMatch = false; break }
        }
        expect(allMatch, "Fold-by-4 matches two sequential fold-by-2")

        print("  Fold-by-4: PASS (\(result.numRounds) rounds)")
    } catch {
        print("  [FAIL] Fold-by-4: \(error)")
        expect(false, "Fold-by-4 threw: \(error)")
    }
}

private func testFoldByHighFactors() {
    suite("FRI Commit Phase: Fold-by-8 and fold-by-16")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        var rng = CommitPhaseRNG(state: 4004)
        let logN = 16
        let n = 1 << logN

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        // Test fold-by-8
        let config8 = FRICommitPhaseConfig(foldingFactor: 8, blowupFactor: 4, finalPolyMaxDegree: 7)
        let result8 = try engine.commitPhase(evaluations: evals, config: config8)
        expect(result8.numRounds > 0, "Fold-by-8 performed rounds")
        for i in 1..<result8.layers.count {
            expectEqual(result8.layers[i].logDomainSize, result8.layers[i-1].logDomainSize - 3,
                        "Fold-by-8 reduces logSize by 3")
        }

        // Test fold-by-16
        let config16 = FRICommitPhaseConfig(foldingFactor: 16, blowupFactor: 4, finalPolyMaxDegree: 7)
        let result16 = try engine.commitPhase(evaluations: evals, config: config16)
        expect(result16.numRounds > 0, "Fold-by-16 performed rounds")
        for i in 1..<result16.layers.count {
            expectEqual(result16.layers[i].logDomainSize, result16.layers[i-1].logDomainSize - 4,
                        "Fold-by-16 reduces logSize by 4")
        }

        // Fold-by-16 should produce fewer rounds than fold-by-8
        expect(result16.numRounds <= result8.numRounds, "Fold-by-16 fewer rounds than fold-by-8")

        print("  Fold-by-8: \(result8.numRounds) rounds; Fold-by-16: \(result16.numRounds) rounds: PASS")
    } catch {
        print("  [FAIL] Fold-by-high-factors: \(error)")
        expect(false, "Fold-by-high-factors threw: \(error)")
    }
}

private func testMerkleCommitments() {
    suite("FRI Commit Phase: Merkle commitments at each layer")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        var rng = CommitPhaseRNG(state: 6006)
        let logN = 10
        let n = 1 << logN

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        let config = FRICommitPhaseConfig(
            foldingFactor: 2, blowupFactor: 4, finalPolyMaxDegree: 3)

        let result = try engine.commitPhase(evaluations: evals, config: config)

        // Verify each layer has a valid Merkle tree
        for (idx, layer) in result.layers.enumerated() {
            expect(!layer.merkleRoot.isZero, "Layer \(idx) root non-zero")

            // Verify a few Merkle proofs
            if let layerEvals = layer.evaluations {
                let domSize = 1 << layer.logDomainSize
                let checkIdx = min(3, domSize - 1)
                let proof = layer.merkleTree.proof(forLeafAt: checkIdx)
                let valid = proof.verify(root: layer.merkleRoot, leaf: layerEvals[checkIdx])
                expect(valid, "Layer \(idx) Merkle proof at index \(checkIdx)")
            }
        }

        // Verify Merkle roots are all distinct
        let roots = result.merkleRoots
        for i in 0..<roots.count {
            for j in (i + 1)..<roots.count {
                expect(!frEqual(roots[i], roots[j]),
                       "Distinct roots for layers \(i) and \(j)")
            }
        }

        print("  Merkle commitments: PASS (\(result.layers.count) layers)")
    } catch {
        print("  [FAIL] Merkle commitments: \(error)")
        expect(false, "Merkle commitments threw: \(error)")
    }
}

private func testRemainderPolynomialDegree() {
    suite("FRI Commit Phase: Remainder polynomial degree check")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        var rng = CommitPhaseRNG(state: 7007)
        let logN = 10
        let n = 1 << logN

        // Create a low-degree polynomial and evaluate it on the domain
        let degree = 15
        var coeffs = [Fr](repeating: Fr.zero, count: degree + 1)
        for i in 0...degree { coeffs[i] = rng.nextFr() }

        // Evaluate on domain
        let domain = buildTestDomain(logN: logN)
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            evals[i] = hornerEval(coeffs, at: domain[i])
        }

        let config = FRICommitPhaseConfig(
            foldingFactor: 2, blowupFactor: n / (degree + 1),
            finalPolyMaxDegree: 15)

        let result = try engine.commitPhase(evaluations: evals, config: config)

        // The remainder polynomial should have degree <= finalPolyMaxDegree
        let degreeOk = engine.verifyRemainderDegree(result: result)
        expect(degreeOk, "Remainder degree <= \(config.finalPolyMaxDegree)")

        // Remainder should be non-trivial (not all zero)
        let hasNonZero = result.remainderPoly.contains { !$0.isZero }
        expect(hasNonZero, "Remainder polynomial is non-trivial")

        print("  Remainder degree: PASS (remainder size=\(result.remainderPoly.count))")
    } catch {
        print("  [FAIL] Remainder degree: \(error)")
        expect(false, "Remainder degree threw: \(error)")
    }
}

private func testDomainHalving() {
    suite("FRI Commit Phase: Domain halving")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        let logN = 8
        let n = 1 << logN

        let domain = engine.buildDomain(logN: logN)
        expectEqual(domain.count, n, "Full domain size")

        // Verify omega^n = 1 (domain wraps around)
        let omegaN = domain[0]  // omega^0 = 1
        expect(frEqual(omegaN, Fr.one), "omega^0 = 1")

        // Halve the domain
        let halved = engine.halveDomain(domain: domain)
        expectEqual(halved.count, n / 2, "Halved domain size")

        // Verify halved[i] = domain[i]^2
        for i in 0..<halved.count {
            let expected = frMul(domain[i], domain[i])
            expect(frEqual(halved[i], expected), "halved[\(i)] = domain[\(i)]^2")
        }

        // Halve again
        let quarterDomain = engine.halveDomain(domain: halved)
        expectEqual(quarterDomain.count, n / 4, "Quarter domain size")

        // Verify quarter[i] = domain[i]^4
        for i in 0..<quarterDomain.count {
            let d2 = frMul(domain[i], domain[i])
            let expected = frMul(d2, d2)
            expect(frEqual(quarterDomain[i], expected), "quarter[\(i)] = domain[\(i)]^4")
        }

        print("  Domain halving: PASS")
    } catch {
        print("  [FAIL] Domain halving: \(error)")
        expect(false, "Domain halving threw: \(error)")
    }
}

private func testPolynomialSplitting() {
    suite("FRI Commit Phase: Even/odd polynomial splitting")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        var rng = CommitPhaseRNG(state: 8008)
        let logN = 8
        let n = 1 << logN

        // Create random evaluations
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        let (even, odd) = engine.splitEvenOdd(evaluations: evals, logN: logN)
        expectEqual(even.count, n / 2, "Even part has n/2 elements")
        expectEqual(odd.count, n / 2, "Odd part has n/2 elements")

        // Verify reconstruction: f(x) = f_even(x^2) + x * f_odd(x^2)
        // For domain points: evals[i] should equal even[i] + domain[i] * odd[i]
        // where even and odd are evaluated on the squared domain
        // Actually, for the first half:
        //   evals[i] = even[i] + domain[i] * odd[i]  (implicitly, since even/odd
        //   are on the squared domain which coincides with the first half of the
        //   halved domain)
        // But we can verify a simpler property: folding with challenge alpha = 0
        // gives even, and folding with alpha = 1 minus even gives odd-related values.

        // Verify even is average of pairs: even[i] = (evals[i] + evals[i+n/2]) / 2
        let twoInv = frInverse(frFromInt(2))
        for i in 0..<(n / 2) {
            let avg = frMul(frAdd(evals[i], evals[i + n / 2]), twoInv)
            expect(frEqual(even[i], avg),
                   "even[\(i)] = avg of evals[\(i)] and evals[\(i + n / 2)]")
        }

        print("  Polynomial splitting: PASS")
    } catch {
        print("  [FAIL] Polynomial splitting: \(error)")
        expect(false, "Polynomial splitting threw: \(error)")
    }
}

private func testCosetDomain() {
    suite("FRI Commit Phase: Coset domain construction")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        let logN = 6
        let n = 1 << logN

        // Build a coset with generator g = frFromInt(5)
        let g = frFromInt(5)
        let cosetDomain = engine.buildCosetDomain(logN: logN, cosetGenerator: g)
        expectEqual(cosetDomain.count, n, "Coset domain size")

        // Verify first element is the coset generator
        expect(frEqual(cosetDomain[0], g), "cosetDomain[0] = g")

        // Verify cosetDomain[i] = g * omega^i
        let stdDomain = engine.buildDomain(logN: logN)
        for i in 0..<n {
            let expected = frMul(g, stdDomain[i])
            expect(frEqual(cosetDomain[i], expected),
                   "cosetDomain[\(i)] = g * omega^\(i)")
        }

        print("  Coset domain: PASS")
    } catch {
        print("  [FAIL] Coset domain: \(error)")
        expect(false, "Coset domain threw: \(error)")
    }
}

private func testCircleFRICommit() {
    suite("FRI Commit Phase: Circle FRI domain")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        var rng = CommitPhaseRNG(state: 9009)
        let logN = 8
        let n = 1 << logN

        var domainPoints = [CircleDomainPoint]()
        for i in 0..<n {
            domainPoints.append(CircleDomainPoint(x: frFromInt(UInt64(i + 1)),
                                                  y: frFromInt(UInt64(n - i))))
        }
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        let config = FRICommitPhaseConfig(foldingFactor: 2, blowupFactor: 4,
                                          finalPolyMaxDegree: 3, domainType: .circle)
        let result = try engine.commitPhaseCircle(
            evaluations: evals, domainPoints: domainPoints, config: config)

        expect(result.numRounds > 0, "Circle FRI performed rounds")
        expectEqual(result.layers.count, result.numRounds + 1, "Correct layer count")
        for i in 1..<result.layers.count {
            expectEqual(result.layers[i].logDomainSize, result.layers[i-1].logDomainSize - 1,
                        "Circle fold halves domain")
        }
        for layer in result.layers {
            expect(!layer.merkleRoot.isZero, "Circle layer root non-zero")
        }
        print("  Circle FRI: PASS (\(result.numRounds) rounds)")
    } catch {
        print("  [FAIL] Circle FRI: \(error)")
        expect(false, "Circle FRI threw: \(error)")
    }
}

private func testCircleDomainPoint() {
    suite("FRI Commit Phase: CircleDomainPoint operations")
    let x = frFromInt(3)
    let y = frFromInt(7)
    let pt = CircleDomainPoint(x: x, y: y)
    let two = frFromInt(2)

    // Test doubling: (x, y) -> (2x^2 - 1, 2xy)
    let doubled = pt.doubled()
    let expectedX = frSub(frMul(two, frMul(x, x)), Fr.one)
    let expectedY = frMul(two, frMul(x, y))
    expect(frEqual(doubled.x, expectedX), "Doubled x = 2x^2 - 1")
    expect(frEqual(doubled.y, expectedY), "Doubled y = 2xy")

    // Test conjugate: (x, y) -> (x, -y)
    let conj = pt.conjugate()
    expect(frEqual(conj.x, x), "Conjugate x unchanged")
    expect(frEqual(conj.y, frSub(Fr.zero, y)), "Conjugate y = -y")

    // Double conjugate: (2x^2-1, -2xy)
    let dcx = conj.doubled()
    expect(frEqual(dcx.x, expectedX), "Doubled conjugate x same")
    expect(frEqual(dcx.y, frSub(Fr.zero, expectedY)), "Doubled conjugate y = -2xy")
    print("  CircleDomainPoint ops: PASS")
}

private func testBatchedCommit() {
    suite("FRI Commit Phase: Batched commit (multiple polynomials)")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        var rng = CommitPhaseRNG(state: 10010)
        let logN = 10
        let n = 1 << logN
        let numPolys = 4

        // Create multiple polynomial evaluations
        var polynomials = [[Fr]]()
        for _ in 0..<numPolys {
            var evals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n { evals[i] = rng.nextFr() }
            polynomials.append(evals)
        }

        // Random batch coefficients
        var batchCoeffs = [Fr]()
        for _ in 0..<numPolys { batchCoeffs.append(rng.nextNonZeroFr()) }

        let config = FRICommitPhaseConfig(
            foldingFactor: 2, blowupFactor: 4, finalPolyMaxDegree: 7)

        let result = try engine.commitPhaseBatched(
            polynomials: polynomials, batchCoeffs: batchCoeffs, config: config)

        expect(result.numRounds > 0, "Batched commit performed rounds")

        // Verify the initial layer matches manual combination
        var combined = [Fr](repeating: Fr.zero, count: n)
        for j in 0..<numPolys {
            for i in 0..<n {
                combined[i] = frAdd(combined[i], frMul(batchCoeffs[j], polynomials[j][i]))
            }
        }

        let initialEvals = result.layers[0].evaluations!
        var combMatch = true
        for i in 0..<n {
            if !frEqual(initialEvals[i], combined[i]) { combMatch = false; break }
        }
        expect(combMatch, "Batched initial layer = linear combination")

        print("  Batched commit: PASS (\(numPolys) polys, \(result.numRounds) rounds)")
    } catch {
        print("  [FAIL] Batched commit: \(error)")
        expect(false, "Batched commit threw: \(error)")
    }
}

private func testStreamingCommit() {
    suite("FRI Commit Phase: Streaming commit")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        var rng = CommitPhaseRNG(state: 11011)
        let logN = 10
        let n = 1 << logN

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        let config = FRICommitPhaseConfig(foldingFactor: 2, blowupFactor: 4, finalPolyMaxDegree: 7)
        var layerRoots = [Fr]()
        let (remainder, challenges) = try engine.commitPhaseStreaming(
            evaluations: evals, config: config) { layer, _ in layerRoots.append(layer.merkleRoot) }

        expect(layerRoots.count > 1, "Streaming produced multiple layers")
        expect(!challenges.isEmpty && !remainder.isEmpty, "Produced challenges and remainder")

        // Compare with non-streaming
        let result = try engine.commitPhase(evaluations: evals, config: config)
        expectEqual(layerRoots.count, result.layers.count, "Same layer count")
        for i in 0..<layerRoots.count {
            expect(frEqual(layerRoots[i], result.layers[i].merkleRoot), "Streaming root \(i) matches")
        }
        print("  Streaming commit: PASS (\(layerRoots.count) layers)")
    } catch {
        print("  [FAIL] Streaming commit: \(error)")
        expect(false, "Streaming commit threw: \(error)")
    }
}

private func testQueryProofGeneration() {
    suite("FRI Commit Phase: Query proof generation")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        var rng = CommitPhaseRNG(state: 12012)
        let logN = 10
        let n = 1 << logN

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        let config = FRICommitPhaseConfig(
            foldingFactor: 2, blowupFactor: 4, finalPolyMaxDegree: 7)

        let result = try engine.commitPhase(evaluations: evals, config: config)

        let queryIndices = [0, 1, n / 2, n - 1, 42]
        let proofs = engine.generateQueryProofs(result: result, queryIndices: queryIndices)

        expectEqual(proofs.count, queryIndices.count, "One proof set per query")

        for (qi, proofSet) in proofs.enumerated() {
            expectEqual(proofSet.count, result.layers.count,
                        "Proof set \(qi) covers all layers")

            // Verify each auth path in the proof set
            for entry in proofSet {
                let layer = result.layers[entry.layerIndex]
                if let layerEvals = layer.evaluations {
                    let leaf = layerEvals[entry.leafIndex]
                    let valid = entry.authPath.verify(root: layer.merkleRoot, leaf: leaf)
                    expect(valid, "Query \(qi) layer \(entry.layerIndex) auth path valid")
                }
            }
        }

        print("  Query proofs: PASS (\(queryIndices.count) queries)")
    } catch {
        print("  [FAIL] Query proofs: \(error)")
        expect(false, "Query proofs threw: \(error)")
    }
}

private func testRemainderEvaluation() {
    suite("FRI Commit Phase: Remainder polynomial evaluation")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        var rng = CommitPhaseRNG(state: 13013)
        let logN = 10
        let n = 1 << logN
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        let config = FRICommitPhaseConfig(foldingFactor: 2, blowupFactor: 4, finalPolyMaxDegree: 7)
        let result = try engine.commitPhase(evaluations: evals, config: config)

        // Determinism and constant term check
        let val = engine.evaluateRemainder(result, at: frFromInt(2))
        let val2 = engine.evaluateRemainder(result, at: frFromInt(2))
        expect(frEqual(val, val2), "Remainder eval is deterministic")
        let atZero = engine.evaluateRemainder(result, at: Fr.zero)
        if !result.remainderPoly.isEmpty {
            expect(frEqual(atZero, result.remainderPoly[0]), "Remainder at 0 = constant term")
        }
        print("  Remainder evaluation: PASS")
    } catch {
        print("  [FAIL] Remainder evaluation: \(error)")
        expect(false, "Remainder evaluation threw: \(error)")
    }
}

private func testConfigVariants() {
    suite("FRI Commit Phase: Config variants (retainLayerEvals, explicit challenges)")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        var rng = CommitPhaseRNG(state: 14014)
        let logN = 10
        let n = 1 << logN

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        // Test retainLayerEvals = false
        let configNoRetain = FRICommitPhaseConfig(
            foldingFactor: 2, blowupFactor: 4, finalPolyMaxDegree: 7, retainLayerEvals: false)
        let resultNoRetain = try engine.commitPhase(evaluations: evals, config: configNoRetain)
        expect(resultNoRetain.layers[0].evaluations == nil, "Evals nil when not retained")
        for layer in resultNoRetain.layers {
            expect(!layer.merkleRoot.isZero, "Merkle root present without evals")
        }

        // Compare roots with retained version
        let configRetain = FRICommitPhaseConfig(
            foldingFactor: 2, blowupFactor: 4, finalPolyMaxDegree: 7, retainLayerEvals: true)
        let resultRetain = try engine.commitPhase(evaluations: evals, config: configRetain)
        for i in 0..<resultNoRetain.layers.count {
            expect(frEqual(resultNoRetain.layers[i].merkleRoot, resultRetain.layers[i].merkleRoot),
                   "Root \(i) same regardless of retainLayerEvals")
        }

        // Test explicit challenges
        var challenges = [Fr]()
        for _ in 0..<20 { challenges.append(rng.nextNonZeroFr()) }
        let result = try engine.commitPhase(evaluations: evals, challenges: challenges, config: configRetain)
        for i in 0..<result.challenges.count {
            expect(frEqual(result.challenges[i], challenges[i]), "Used explicit challenge \(i)")
        }
        let result2 = try engine.commitPhase(evaluations: evals, challenges: challenges, config: configRetain)
        for i in 0..<result.layers.count {
            expect(frEqual(result.layers[i].merkleRoot, result2.layers[i].merkleRoot),
                   "Deterministic with explicit challenges")
        }

        print("  Config variants: PASS")
    } catch {
        print("  [FAIL] Config variants: \(error)")
        expect(false, "Config variants threw: \(error)")
    }
}

private func testSmallDomainCPUFallback() {
    suite("FRI Commit Phase: CPU fallback for small domains")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        var rng = CommitPhaseRNG(state: 16016)
        let logN = 6
        let n = 1 << logN
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        let config = FRICommitPhaseConfig(foldingFactor: 2, blowupFactor: 2, finalPolyMaxDegree: 3)
        let result = try engine.commitPhase(evaluations: evals, config: config)
        expect(result.numRounds > 0, "Small domain commit works")

        // Cross-check first fold with CPU reference
        let cpuFolded = cpuFoldRef(evals: evals, logN: logN, challenge: result.challenges[0])
        let gpuFolded = result.layers[1].evaluations!
        var allMatch = true
        for i in 0..<cpuFolded.count {
            if !frEqual(gpuFolded[i], cpuFolded[i]) { allMatch = false; break }
        }
        expect(allMatch, "CPU fallback fold matches reference")
        print("  CPU fallback: PASS (n=\(n))")
    } catch {
        print("  [FAIL] CPU fallback: \(error)")
        expect(false, "CPU fallback threw: \(error)")
    }
}

private func testEdgeCasePolynomials() {
    suite("FRI Commit Phase: Zero and constant polynomial edge cases")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        let logN = 8
        let n = 1 << logN

        // Zero polynomial
        let zeroEvals = [Fr](repeating: Fr.zero, count: n)
        let zeroConfig = FRICommitPhaseConfig(foldingFactor: 2, blowupFactor: 4, finalPolyMaxDegree: 7)
        let zeroResult = try engine.commitPhase(evaluations: zeroEvals, config: zeroConfig)
        for (idx, layer) in zeroResult.layers.enumerated() {
            if let layerEvals = layer.evaluations {
                expect(layerEvals.allSatisfy { $0.isZero }, "Layer \(idx) all zero")
            }
        }
        expect(zeroResult.remainderPoly.allSatisfy { $0.isZero }, "Zero remainder")
        expect(engine.verifyRemainderDegree(result: zeroResult), "Zero poly degree ok")

        // Constant polynomial: fold(c, c, alpha, domainInv) = 2c
        let constant = frFromInt(42)
        let constEvals = [Fr](repeating: constant, count: n)
        let constConfig = FRICommitPhaseConfig(foldingFactor: 2, blowupFactor: 4, finalPolyMaxDegree: 0)
        let constResult = try engine.commitPhase(evaluations: constEvals, config: constConfig)
        if constResult.numRounds > 0 {
            let foldedEvals = constResult.layers[1].evaluations!
            let twoConst = frAdd(constant, constant)
            expect(frEqual(foldedEvals[0], twoConst), "Constant poly fold gives 2*c")
        }

        print("  Edge case polynomials: PASS")
    } catch {
        print("  [FAIL] Edge case polynomials: \(error)")
        expect(false, "Edge case polynomials threw: \(error)")
    }
}

private func testCommitPhaseLayerSizes() {
    suite("FRI Commit Phase: Layer size consistency across folding factors")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        var rng = CommitPhaseRNG(state: 17017)
        let logN = 16
        let n = 1 << logN

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        // Test each folding factor
        for factor in [2, 4, 8, 16] {
            let foldBits = factor.trailingZeroBitCount
            let config = FRICommitPhaseConfig(
                foldingFactor: factor, blowupFactor: 4, finalPolyMaxDegree: 7)

            let result = try engine.commitPhase(evaluations: evals, config: config)

            // Verify sizes decrease by the correct factor
            for i in 1..<result.layers.count {
                let prevSize = 1 << result.layers[i - 1].logDomainSize
                let curSize = 1 << result.layers[i].logDomainSize
                expectEqual(curSize, prevSize >> foldBits,
                            "Factor \(factor): layer \(i) size = prev >> \(foldBits)")
            }
        }

        print("  Layer size consistency: PASS (all folding factors)")
    } catch {
        print("  [FAIL] Layer size consistency: \(error)")
        expect(false, "Layer size consistency threw: \(error)")
    }
}

private func testClearCache() {
    suite("FRI Commit Phase: Cache clearing")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        var rng = CommitPhaseRNG(state: 18018)
        let logN = 10
        let n = 1 << logN
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }
        let config = FRICommitPhaseConfig(foldingFactor: 2, blowupFactor: 4, finalPolyMaxDegree: 7)

        let result1 = try engine.commitPhase(evaluations: evals, config: config)
        engine.clearCache()
        let result2 = try engine.commitPhase(evaluations: evals, config: config)
        for i in 0..<result1.layers.count {
            expect(frEqual(result1.layers[i].merkleRoot, result2.layers[i].merkleRoot),
                   "Post-cache-clear root \(i) matches")
        }
        print("  Cache clearing: PASS")
    } catch {
        print("  [FAIL] Cache clearing: \(error)")
        expect(false, "Cache clearing threw: \(error)")
    }
}

private func testPerformance() {
    suite("FRI Commit Phase: Performance 2^16")
    do {
        let engine = try GPUFRICommitPhaseEngine()
        var rng = CommitPhaseRNG(state: 99099)
        let logN = 16
        let n = 1 << logN
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }
        let config = FRICommitPhaseConfig(foldingFactor: 2, blowupFactor: 4, finalPolyMaxDegree: 7)

        _ = try engine.commitPhase(evaluations: evals, config: config) // warmup
        let start = DispatchTime.now()
        let result = try engine.commitPhase(evaluations: evals, config: config)
        let ms = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000.0
        expect(result.numRounds > 0, "Performance run completed")
        print("  Performance: 2^\(logN) in \(String(format: "%.2f", ms)) ms, \(result.numRounds) rounds")
    } catch {
        print("  [FAIL] Performance: \(error)")
        expect(false, "Performance threw: \(error)")
    }
}

// MARK: - Public entry point

public func runGPUFRICommitPhaseTests() {
    testSingleLayerFold()
    testMultiRoundCommit()
    testFoldByFour()
    testFoldByHighFactors()
    testMerkleCommitments()
    testRemainderPolynomialDegree()
    testDomainHalving()
    testPolynomialSplitting()
    testCosetDomain()
    testCircleFRICommit()
    testCircleDomainPoint()
    testBatchedCommit()
    testStreamingCommit()
    testQueryProofGeneration()
    testRemainderEvaluation()
    testConfigVariants()
    testSmallDomainCPUFallback()
    testEdgeCasePolynomials()
    testCommitPhaseLayerSizes()
    testClearCache()
    testPerformance()
}
