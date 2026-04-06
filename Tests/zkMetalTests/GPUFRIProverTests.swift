// GPU FRI Prover Engine tests
// Validates commit phase layer count, query response validity, folding correctness,
// small polynomial proof, and final poly degree check.
// Run: swift build && .build/debug/zkMetalTests

import zkMetal
import Foundation

// MARK: - Test RNG

private struct ProverRNG {
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

// MARK: - Helpers

/// Build evaluations of a polynomial with known degree by evaluating at roots of unity.
/// coeffs has (degree+1) entries; we evaluate on domain of size n = 2^logN.
private func evaluatePoly(coeffs: [Fr], logN: Int) -> [Fr] {
    let n = 1 << logN
    let invTwiddles = precomputeInverseTwiddles(logN: logN)
    // omega = 1 / invTwiddle[1]
    let omega = frInverse(invTwiddles[1])

    var evals = [Fr](repeating: Fr.zero, count: n)
    var omegaPow = Fr.one  // omega^i for evaluation point i
    for i in 0..<n {
        // Horner evaluation: p(x) = c0 + x*(c1 + x*(c2 + ...))
        var val = Fr.zero
        for j in stride(from: coeffs.count - 1, through: 0, by: -1) {
            val = frAdd(frMul(val, omegaPow), coeffs[j])
        }
        evals[i] = val
        omegaPow = frMul(omegaPow, omega)
    }
    return evals
}

// MARK: - Tests

private func testCommitPhaseLayerCount() {
    suite("FRI Prover: Commit phase layer count")
    do {
        let engine = try GPUFRIProverEngine()
        var rng = ProverRNG(state: 42)
        let logN = 10
        let n = 1 << logN

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        let config = GPUFRIConfig(foldingFactor: 2, blowupFactor: 4,
                               numQueries: 8, finalPolyMaxDegree: 7)

        // Provide explicit challenges for determinism
        var challenges = [Fr]()
        for _ in 0..<logN { challenges.append(rng.nextFr()) }

        let commitment = try engine.commit(evaluations: evals,
                                            challenges: challenges,
                                            config: config)

        // With foldingFactor=2, blowup=4, finalMaxDeg=7:
        // target size = (7+1)*4 = 32 = 2^5
        // rounds = log2(1024/32) = 5
        // layers = rounds + 1 = 6
        let expectedRounds = 5
        let expectedLayers = expectedRounds + 1
        expect(commitment.layers.count == expectedLayers,
               "Layer count is \(expectedLayers), got \(commitment.layers.count)")
        expect(commitment.challenges.count == expectedRounds,
               "Challenge count is \(expectedRounds)")

        // Verify layer sizes halve each round
        for i in 1..<commitment.layers.count {
            let prevSize = commitment.layers[i - 1].evaluations.count
            let curSize = commitment.layers[i].evaluations.count
            expect(curSize == prevSize / 2,
                   "Layer \(i) size \(curSize) == \(prevSize)/2")
        }

        // Verify final layer size
        let finalSize = commitment.layers.last!.evaluations.count
        expect(finalSize == 32, "Final layer size is 32, got \(finalSize)")

        print("  Commit phase layers: \(commitment.layers.count) (expected \(expectedLayers))")
    } catch {
        print("  [FAIL] Commit layer count: \(error)")
        expect(false, "Commit phase threw: \(error)")
    }
}

private func testCommitPhaseLayerCountFold4() {
    suite("FRI Prover: Commit phase with folding factor 4")
    do {
        let engine = try GPUFRIProverEngine()
        var rng = ProverRNG(state: 55)
        let logN = 12
        let n = 1 << logN

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        let config = GPUFRIConfig(foldingFactor: 4, blowupFactor: 4,
                               numQueries: 8, finalPolyMaxDegree: 3)

        var challenges = [Fr]()
        for _ in 0..<(logN / 2) { challenges.append(rng.nextFr()) }

        let commitment = try engine.commit(evaluations: evals,
                                            challenges: challenges,
                                            config: config)

        // foldingFactor=4 folds 2 bits per round
        // target size = (3+1)*4 = 16 = 2^4
        // rounds = (12 - 4) / 2 = 4
        // layers = 4 + 1 = 5
        let expectedRounds = 4
        expect(commitment.layers.count == expectedRounds + 1,
               "Fold-4 layer count correct")

        // Each layer should be 1/4 the previous
        for i in 1..<commitment.layers.count {
            let prevSize = commitment.layers[i - 1].evaluations.count
            let curSize = commitment.layers[i].evaluations.count
            expect(curSize == prevSize / 4,
                   "Layer \(i) size \(curSize) == \(prevSize)/4")
        }

        print("  Fold-4 layers: \(commitment.layers.count)")
    } catch {
        print("  [FAIL] Fold-4 commit: \(error)")
        expect(false, "Fold-4 commit threw: \(error)")
    }
}

private func testQueryResponseValidity() {
    suite("FRI Prover: Query response validity")
    do {
        let engine = try GPUFRIProverEngine()
        var rng = ProverRNG(state: 77)
        let logN = 10
        let n = 1 << logN

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        let config = GPUFRIConfig(foldingFactor: 2, blowupFactor: 4,
                               numQueries: 8, finalPolyMaxDegree: 7)

        var challenges = [Fr]()
        for _ in 0..<logN { challenges.append(rng.nextFr()) }

        let commitment = try engine.commit(evaluations: evals,
                                            challenges: challenges,
                                            config: config)
        let queryIndices = [0, 42, 100, 511, 512, 700, 900, 1023]
        let responses = try engine.query(commitment: commitment,
                                          queryIndices: queryIndices)

        expect(responses.count == 8, "Got 8 query responses")

        // Verify each query response
        var allValid = true
        for resp in responses {
            // Each response should have layerProofs for each layer transition
            let expectedProofs = commitment.layers.count - 1
            if resp.layerProofs.count != expectedProofs {
                allValid = false
                print("  Wrong proof count: \(resp.layerProofs.count) != \(expectedProofs)")
                break
            }

            // Verify Merkle paths
            let valid = GPUFRIProverEngine.verifyQueryResponse(
                response: resp, commitment: commitment)
            if !valid {
                allValid = false
                print("  Query \(resp.queryIndex) failed verification")
                break
            }
        }
        expect(allValid, "All query responses valid")
        print("  Query responses: \(allValid ? "PASS" : "FAIL") (\(responses.count) queries)")
    } catch {
        print("  [FAIL] Query validity: \(error)")
        expect(false, "Query validity threw: \(error)")
    }
}

private func testFoldingCorrectness() {
    suite("FRI Prover: Folding correctness across layers")
    do {
        let engine = try GPUFRIProverEngine()
        var rng = ProverRNG(state: 100)
        let logN = 8
        let n = 1 << logN

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        let config = GPUFRIConfig(foldingFactor: 2, blowupFactor: 2,
                               numQueries: 4, finalPolyMaxDegree: 3)

        var challenges = [Fr]()
        for _ in 0..<logN { challenges.append(rng.nextFr()) }

        let commitment = try engine.commit(evaluations: evals,
                                            challenges: challenges,
                                            config: config)

        // Use the GPU fold engine to verify fold correctness independently
        let foldEngine = try GPUFRIFoldEngine()
        let layers = try foldEngine.foldAllRounds(evals: evals, logN: logN,
                                                    challenges: Array(challenges.prefix(commitment.layers.count - 1)))

        // Compare layer evaluations
        var allMatch = true
        for i in 0..<commitment.layers.count {
            let commitEvals = commitment.layers[i].evaluations
            let foldEvals = layers[i]
            if commitEvals.count != foldEvals.count {
                allMatch = false
                print("  Size mismatch at layer \(i): \(commitEvals.count) vs \(foldEvals.count)")
                break
            }
            for j in 0..<commitEvals.count {
                if !frEqual(commitEvals[j], foldEvals[j]) {
                    allMatch = false
                    print("  Value mismatch at layer \(i), index \(j)")
                    break
                }
            }
            if !allMatch { break }
        }
        expect(allMatch, "FRI commit layers match independent fold engine")
        print("  Folding correctness: \(allMatch ? "PASS" : "FAIL")")
    } catch {
        print("  [FAIL] Folding correctness: \(error)")
        expect(false, "Folding correctness threw: \(error)")
    }
}

private func testSmallPolynomialProof() {
    suite("FRI Prover: Small polynomial end-to-end proof")
    do {
        let engine = try GPUFRIProverEngine()

        // Polynomial of degree 3: p(x) = 1 + 2x + 3x^2 + 4x^3
        let coeffs: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let degree = 3

        // Evaluate on domain of size 2^6 = 64 (blowup = 64/4 = 16)
        let logN = 6
        let evals = evaluatePoly(coeffs: coeffs, logN: logN)

        let config = GPUFRIConfig(foldingFactor: 2, blowupFactor: 4,
                               numQueries: 4, finalPolyMaxDegree: 3)

        let proof = try engine.prove(evaluations: evals, config: config)

        // Verify we got layers
        expect(proof.commitment.layers.count >= 2, "At least 2 layers")

        // Verify we got query responses
        expect(proof.queryResponses.count == config.numQueries,
               "Got \(config.numQueries) query responses")

        // Verify Merkle paths are valid
        var allQueriesValid = true
        for resp in proof.queryResponses {
            let valid = GPUFRIProverEngine.verifyQueryResponse(
                response: resp, commitment: proof.commitment)
            if !valid {
                allQueriesValid = false
                break
            }
        }
        expect(allQueriesValid, "All queries verify against commitment")

        // Final polynomial should exist and be non-empty
        expect(proof.commitment.finalPoly.count > 0, "Final poly is non-empty")

        print("  Small poly proof: PASS (degree=\(degree), n=\(1 << logN))")
    } catch {
        print("  [FAIL] Small polynomial proof: \(error)")
        expect(false, "Small poly proof threw: \(error)")
    }
}

private func testFinalPolyDegreeCheck() {
    suite("FRI Prover: Final polynomial degree check")
    do {
        // Test the static degree checker
        let low = [frFromInt(1), frFromInt(2), frFromInt(3), Fr.zero, Fr.zero, Fr.zero, Fr.zero, Fr.zero]
        expect(GPUFRIProverEngine.verifyFinalPolyDegree(finalPoly: low, maxDegree: 7),
               "Poly with high coeffs zero passes degree 7 check")
        expect(GPUFRIProverEngine.verifyFinalPolyDegree(finalPoly: low, maxDegree: 2),
               "Poly [1,2,3,0,0,...] passes degree 2 check")
        expect(!GPUFRIProverEngine.verifyFinalPolyDegree(finalPoly: low, maxDegree: 1),
               "Poly [1,2,3,0,...] fails degree 1 check")

        let high = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        expect(GPUFRIProverEngine.verifyFinalPolyDegree(finalPoly: high, maxDegree: 3),
               "Poly degree 3 passes maxDegree=3")
        expect(!GPUFRIProverEngine.verifyFinalPolyDegree(finalPoly: high, maxDegree: 2),
               "Poly degree 3 fails maxDegree=2")

        // Empty / single
        expect(GPUFRIProverEngine.verifyFinalPolyDegree(finalPoly: [], maxDegree: 0),
               "Empty poly passes degree 0")
        expect(GPUFRIProverEngine.verifyFinalPolyDegree(finalPoly: [Fr.one], maxDegree: 0),
               "Constant poly passes degree 0")

        print("  Final poly degree check: PASS")
    } catch {
        print("  [FAIL] Final poly degree: \(error)")
        expect(false, "Final poly degree threw: \(error)")
    }
}

private func testMerkleCommitmentsNonTrivial() {
    suite("FRI Prover: Merkle commitments non-trivial")
    do {
        let engine = try GPUFRIProverEngine()
        var rng = ProverRNG(state: 200)
        let logN = 8
        let n = 1 << logN

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        let config = GPUFRIConfig(foldingFactor: 2, blowupFactor: 2,
                               numQueries: 4, finalPolyMaxDegree: 3)

        var challenges = [Fr]()
        for _ in 0..<logN { challenges.append(rng.nextFr()) }

        let commitment = try engine.commit(evaluations: evals,
                                            challenges: challenges,
                                            config: config)

        // All Merkle roots should be non-zero
        var allNonZero = true
        for (i, layer) in commitment.layers.enumerated() {
            if layer.merkleTree.root.isZero {
                allNonZero = false
                print("  Layer \(i) has zero Merkle root")
                break
            }
        }
        expect(allNonZero, "All layer Merkle roots are non-zero")

        // Consecutive roots should differ (folded evals differ)
        var allDiffer = true
        for i in 1..<commitment.layers.count {
            let prev = commitment.layers[i - 1].merkleTree.root
            let curr = commitment.layers[i].merkleTree.root
            if frEqual(prev, curr) {
                allDiffer = false
                print("  Layers \(i-1) and \(i) have same Merkle root")
                break
            }
        }
        expect(allDiffer, "Consecutive layer roots differ")

        print("  Merkle commitments: \(allNonZero && allDiffer ? "PASS" : "FAIL")")
    } catch {
        print("  [FAIL] Merkle commitments: \(error)")
        expect(false, "Merkle commitments threw: \(error)")
    }
}

// MARK: - Public entry point

public func runGPUFRIProverTests() {
    testCommitPhaseLayerCount()
    testCommitPhaseLayerCountFold4()
    testQueryResponseValidity()
    testFoldingCorrectness()
    testSmallPolynomialProof()
    testFinalPolyDegreeCheck()
    testMerkleCommitmentsNonTrivial()
}
