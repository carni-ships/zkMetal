// GPU FRI Verifier Engine tests
// Validates Merkle path verification, folding consistency, final polynomial degree,
// batch verification, degree bound checking, and end-to-end prove+verify round-trip.
// Run: swift build && .build/debug/zkMetalTests

import zkMetal
import Foundation

// MARK: - Test RNG

private struct VerifierRNG {
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
private func evaluatePolyForVerifier(coeffs: [Fr], logN: Int) -> [Fr] {
    let n = 1 << logN
    let invTwiddles = precomputeInverseTwiddles(logN: logN)
    let omega = frInverse(invTwiddles[1])

    var evals = [Fr](repeating: Fr.zero, count: n)
    var omegaPow = Fr.one
    for i in 0..<n {
        var val = Fr.zero
        for j in stride(from: coeffs.count - 1, through: 0, by: -1) {
            val = frAdd(frMul(val, omegaPow), coeffs[j])
        }
        evals[i] = val
        omegaPow = frMul(omegaPow, omega)
    }
    return evals
}

/// Helper: generate a proof using the prover engine with a valid low-degree polynomial.
private func generateProof(logN: Int, config: GPUFRIConfig,
                           seed: UInt64) throws -> GPUFRIProof {
    let prover = try GPUFRIProverEngine()
    var rng = VerifierRNG(state: seed)

    // Generate random coefficients for a polynomial of degree <= finalPolyMaxDegree
    // so that FRI verification (degree check, eval check) can succeed.
    let numCoeffs = config.finalPolyMaxDegree + 1
    var coeffs = [Fr](repeating: Fr.zero, count: numCoeffs)
    for i in 0..<numCoeffs { coeffs[i] = rng.nextFr() }

    let evals = evaluatePolyForVerifier(coeffs: coeffs, logN: logN)
    return try prover.prove(evaluations: evals, config: config)
}

// MARK: - Tests

private func testBasicVerification() {
    suite("FRI Verifier: Basic proof verification")
    do {
        let verifier = try GPUFRIVerifierEngine()
        let config = GPUFRIConfig(foldingFactor: 2, blowupFactor: 4,
                                   numQueries: 8, finalPolyMaxDegree: 7)
        let proof = try generateProof(logN: 10, config: config, seed: 42)

        let valid = try verifier.verify(proof: proof)
        expect(valid, "Basic FRI proof should verify")

        print("  Basic verification: \(valid ? "PASS" : "FAIL")")
    } catch {
        print("  [FAIL] Basic verification: \(error)")
        expect(false, "Basic verification threw: \(error)")
    }
}

private func testDetailedVerification() {
    suite("FRI Verifier: Detailed verification result")
    do {
        let verifier = try GPUFRIVerifierEngine()
        let config = GPUFRIConfig(foldingFactor: 2, blowupFactor: 4,
                                   numQueries: 8, finalPolyMaxDegree: 7)
        let proof = try generateProof(logN: 10, config: config, seed: 55)

        let result = try verifier.verifyDetailed(proof: proof)

        expect(result.isValid, "Detailed result should be valid")
        expect(result.numQueriesChecked == 8, "Checked 8 queries")
        expect(result.merklePathsValid, "Merkle paths should be valid")
        expect(result.foldingConsistencyValid, "Folding consistency should be valid")
        expect(result.finalPolyDegreeValid, "Final poly degree should be valid")
        expect(result.finalPolyEvalsValid, "Final poly evals should be valid")
        expect(result.verificationTimeSeconds > 0, "Verification took some time")
        expect(result.numLayersChecked > 0, "Checked some layers")

        print("  Detailed: valid=\(result.isValid), queries=\(result.numQueriesChecked), " +
              "layers=\(result.numLayersChecked), time=\(String(format: "%.3f", result.verificationTimeSeconds))s")
    } catch {
        print("  [FAIL] Detailed verification: \(error)")
        expect(false, "Detailed verification threw: \(error)")
    }
}

private func testMerklePathVerification() {
    suite("FRI Verifier: Merkle path verification")
    do {
        let _ = try GPUFRIVerifierEngine()  // ensure engine can be created
        let config = GPUFRIConfig(foldingFactor: 2, blowupFactor: 4,
                                   numQueries: 4, finalPolyMaxDegree: 7)
        let proof = try generateProof(logN: 8, config: config, seed: 77)

        // Verify each query's Merkle paths independently
        var allValid = true
        for resp in proof.queryResponses {
            for (layerIdx, layerProof) in resp.layerProofs.enumerated() {
                let layer = proof.commitment.layers[layerIdx]
                let root = layer.merkleTree.root
                let valid = layerProof.authPath.verify(root: root, leaf: layerProof.evaluation)
                if !valid {
                    allValid = false
                    break
                }
            }
            if !allValid { break }
        }
        expect(allValid, "All Merkle paths should verify independently")

        print("  Merkle path verification: \(allValid ? "PASS" : "FAIL")")
    } catch {
        print("  [FAIL] Merkle path verification: \(error)")
        expect(false, "Merkle path verification threw: \(error)")
    }
}

private func testFoldingConsistencyCheck() {
    suite("FRI Verifier: Folding consistency check")
    do {
        let verifier = try GPUFRIVerifierEngine()
        let config = GPUFRIConfig(foldingFactor: 2, blowupFactor: 2,
                                   numQueries: 4, finalPolyMaxDegree: 3)
        let proof = try generateProof(logN: 8, config: config, seed: 100)

        let result = try verifier.verifyDetailed(proof: proof)
        expect(result.foldingConsistencyValid,
               "Folding consistency should pass for honest proof")

        print("  Folding consistency: \(result.foldingConsistencyValid ? "PASS" : "FAIL")")
    } catch {
        print("  [FAIL] Folding consistency: \(error)")
        expect(false, "Folding consistency threw: \(error)")
    }
}

private func testFinalPolyDegreeCheck() {
    suite("FRI Verifier: Final polynomial degree check")
    do {
        let verifier = try GPUFRIVerifierEngine()

        // Test degree checking directly
        let low = [frFromInt(1), frFromInt(2), frFromInt(3),
                   Fr.zero, Fr.zero, Fr.zero, Fr.zero, Fr.zero]
        expect(verifier.verifyFinalPolyDegree(finalPoly: low, maxDegree: 7),
               "Poly with trailing zeros passes degree 7")
        expect(verifier.verifyFinalPolyDegree(finalPoly: low, maxDegree: 2),
               "Poly [1,2,3,0,...] passes degree 2")
        expect(!verifier.verifyFinalPolyDegree(finalPoly: low, maxDegree: 1),
               "Poly [1,2,3,0,...] fails degree 1")

        let high = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        expect(verifier.verifyFinalPolyDegree(finalPoly: high, maxDegree: 3),
               "Poly degree 3 passes maxDegree=3")
        expect(!verifier.verifyFinalPolyDegree(finalPoly: high, maxDegree: 2),
               "Poly degree 3 fails maxDegree=2")

        // Edge cases
        expect(verifier.verifyFinalPolyDegree(finalPoly: [], maxDegree: 0),
               "Empty poly passes degree 0")
        expect(verifier.verifyFinalPolyDegree(finalPoly: [Fr.one], maxDegree: 0),
               "Constant poly passes degree 0")

        print("  Final poly degree check: PASS")
    } catch {
        print("  [FAIL] Final poly degree: \(error)")
        expect(false, "Final poly degree threw: \(error)")
    }
}

private func testEffectiveDegree() {
    suite("FRI Verifier: Effective degree computation")
    do {
        let verifier = try GPUFRIVerifierEngine()

        // Degree 2 poly with trailing zeros
        let poly1 = [frFromInt(5), frFromInt(3), frFromInt(1), Fr.zero, Fr.zero]
        expect(verifier.effectiveDegree(of: poly1) == 2,
               "Effective degree of [5,3,1,0,0] should be 2")

        // Constant
        let poly2 = [frFromInt(7)]
        expect(verifier.effectiveDegree(of: poly2) == 0,
               "Effective degree of [7] should be 0")

        // All zeros (treated as degree 0)
        let poly3 = [Fr.zero, Fr.zero, Fr.zero]
        expect(verifier.effectiveDegree(of: poly3) == 0,
               "Effective degree of [0,0,0] should be 0")

        // Empty
        let poly4: [Fr] = []
        expect(verifier.effectiveDegree(of: poly4) == 0,
               "Effective degree of [] should be 0")

        // Full degree
        let poly5 = [frFromInt(1), frFromInt(2), frFromInt(3)]
        expect(verifier.effectiveDegree(of: poly5) == 2,
               "Effective degree of [1,2,3] should be 2")

        print("  Effective degree: PASS")
    } catch {
        print("  [FAIL] Effective degree: \(error)")
        expect(false, "Effective degree threw: \(error)")
    }
}

private func testSmallPolynomialRoundTrip() {
    suite("FRI Verifier: Small polynomial prove+verify round-trip")
    do {
        let prover = try GPUFRIProverEngine()
        let verifier = try GPUFRIVerifierEngine()

        // p(x) = 1 + 2x + 3x^2 + 4x^3
        let coeffs: [Fr] = [frFromInt(1), frFromInt(2), frFromInt(3), frFromInt(4)]
        let logN = 6
        let evals = evaluatePolyForVerifier(coeffs: coeffs, logN: logN)

        let config = GPUFRIConfig(foldingFactor: 2, blowupFactor: 4,
                                   numQueries: 4, finalPolyMaxDegree: 3)

        let proof = try prover.prove(evaluations: evals, config: config)
        let valid = try verifier.verify(proof: proof)

        expect(valid, "Small polynomial proof should verify")

        print("  Small poly round-trip: \(valid ? "PASS" : "FAIL") (degree=3, n=\(1 << logN))")
    } catch {
        print("  [FAIL] Small poly round-trip: \(error)")
        expect(false, "Small poly round-trip threw: \(error)")
    }
}

private func testLargerProofRoundTrip() {
    suite("FRI Verifier: Larger proof round-trip (2^12)")
    do {
        let prover = try GPUFRIProverEngine()
        let verifier = try GPUFRIVerifierEngine()

        var rng = VerifierRNG(state: 200)
        let logN = 12
        let n = 1 << logN

        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }

        let config = GPUFRIConfig(foldingFactor: 2, blowupFactor: 4,
                                   numQueries: 16, finalPolyMaxDegree: 7)

        let proof = try prover.prove(evaluations: evals, config: config)
        let result = try verifier.verifyDetailed(proof: proof)

        expect(result.isValid, "Larger proof should verify")
        expect(result.numQueriesChecked == 16, "16 queries checked")

        print("  Larger proof: valid=\(result.isValid), " +
              "time=\(String(format: "%.3f", result.verificationTimeSeconds))s")
    } catch {
        print("  [FAIL] Larger proof round-trip: \(error)")
        expect(false, "Larger proof round-trip threw: \(error)")
    }
}

private func testQuickVerification() {
    suite("FRI Verifier: Quick verification mode")
    do {
        let verifier = try GPUFRIVerifierEngine()
        let config = GPUFRIConfig(foldingFactor: 2, blowupFactor: 4,
                                   numQueries: 8, finalPolyMaxDegree: 7)
        let proof = try generateProof(logN: 10, config: config, seed: 333)

        let quickResult = try verifier.quickVerify(proof: proof)
        let fullResult = try verifier.verify(proof: proof)

        expect(quickResult, "Quick verify should pass for honest proof")
        expect(fullResult, "Full verify should pass for honest proof")

        print("  Quick verification: \(quickResult ? "PASS" : "FAIL")")
    } catch {
        print("  [FAIL] Quick verification: \(error)")
        expect(false, "Quick verification threw: \(error)")
    }
}

private func testBatchVerification() {
    suite("FRI Verifier: Batch FRI verification")
    do {
        let verifier = try GPUFRIVerifierEngine()
        let config = GPUFRIConfig(foldingFactor: 2, blowupFactor: 4,
                                   numQueries: 4, finalPolyMaxDegree: 7)

        // Generate 3 independent proofs
        let proof1 = try generateProof(logN: 8, config: config, seed: 111)
        let proof2 = try generateProof(logN: 8, config: config, seed: 222)
        let proof3 = try generateProof(logN: 8, config: config, seed: 333)

        let batchProof = BatchFRIProof(
            proofs: [proof1, proof2, proof3],
            sharedChallenges: [],  // no shared challenges for independent proofs
            batchingCoeffs: [frFromInt(1), frFromInt(2), frFromInt(3)])

        let batchValid = try verifier.verifyBatch(batchProof: batchProof)
        expect(batchValid, "Batch of valid proofs should verify")

        print("  Batch verification: \(batchValid ? "PASS" : "FAIL") (3 proofs)")
    } catch {
        print("  [FAIL] Batch verification: \(error)")
        expect(false, "Batch verification threw: \(error)")
    }
}

private func testDegreeBoundChecking() {
    suite("FRI Verifier: Degree bound checking")
    do {
        let verifier = try GPUFRIVerifierEngine()
        let config = GPUFRIConfig(foldingFactor: 2, blowupFactor: 4,
                                   numQueries: 4, finalPolyMaxDegree: 7)
        let proof = try generateProof(logN: 8, config: config, seed: 500)

        // Generous degree bound should pass
        let generousBound = FRIDegreeBound(
            claimedDegree: 255, blowupFactor: 4, logDomainSize: 8)
        let passGenerous = verifier.checkDegreeBound(proof: proof, degreeBound: generousBound)
        expect(passGenerous, "Generous degree bound should pass")

        // Very tight degree bound (0) should fail if final poly has degree > 0
        let tightBound = FRIDegreeBound(
            claimedDegree: 0, blowupFactor: 4, logDomainSize: 8)
        // This may or may not pass depending on the final poly
        let tightResult = verifier.checkDegreeBound(proof: proof, degreeBound: tightBound)
        let actualDeg = verifier.effectiveDegree(of: proof.commitment.finalPoly)
        let expectedTight = actualDeg <= 0
        expect(tightResult == expectedTight,
               "Tight bound result consistent with actual degree \(actualDeg)")

        print("  Degree bound: generous=\(passGenerous), tight=\(tightResult) (actual deg=\(actualDeg))")
    } catch {
        print("  [FAIL] Degree bound: \(error)")
        expect(false, "Degree bound threw: \(error)")
    }
}

private func testPolynomialEvaluation() {
    suite("FRI Verifier: Polynomial evaluation (Horner)")
    do {
        let verifier = try GPUFRIVerifierEngine()

        // p(x) = 3 + 2x + x^2
        // p(0) = 3, p(1) = 6, p(2) = 11
        let coeffs: [Fr] = [frFromInt(3), frFromInt(2), frFromInt(1)]

        let at0 = verifier.evaluatePolynomial(coeffs, at: Fr.zero)
        expect(frEqual(at0, frFromInt(3)), "p(0) = 3")

        let at1 = verifier.evaluatePolynomial(coeffs, at: Fr.one)
        expect(frEqual(at1, frFromInt(6)), "p(1) = 6")

        let at2 = verifier.evaluatePolynomial(coeffs, at: frFromInt(2))
        expect(frEqual(at2, frFromInt(11)), "p(2) = 11")

        // Empty polynomial
        let empty = verifier.evaluatePolynomial([], at: frFromInt(5))
        expect(frEqual(empty, Fr.zero), "Empty poly = 0")

        // Constant polynomial
        let constant = verifier.evaluatePolynomial([frFromInt(7)], at: frFromInt(99))
        expect(frEqual(constant, frFromInt(7)), "Constant poly = 7")

        print("  Polynomial evaluation: PASS")
    } catch {
        print("  [FAIL] Polynomial evaluation: \(error)")
        expect(false, "Polynomial evaluation threw: \(error)")
    }
}

private func testFoldRecomputation() {
    suite("FRI Verifier: GPU fold recomputation verification")
    do {
        let verifier = try GPUFRIVerifierEngine()
        let foldEngine = try GPUFRIFoldEngine()
        var rng = VerifierRNG(state: 600)

        let logN = 8
        let n = 1 << logN
        var evals = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n { evals[i] = rng.nextFr() }
        let challenge = rng.nextFr()

        // Compute fold with the fold engine
        let foldedLayers = try foldEngine.foldAllRounds(
            evals: evals, logN: logN, challenges: [challenge])
        let expected = foldedLayers[1]  // layer after first fold

        // Verify using the verifier's recomputation
        let match = try verifier.verifyFoldRecomputation(
            evals: evals, challenge: challenge, logN: logN, expectedResult: expected)
        expect(match, "Fold recomputation should match fold engine output")

        // Verify with wrong expected result should fail
        var wrongResult = expected
        wrongResult[0] = frAdd(wrongResult[0], Fr.one)
        let noMatch = try verifier.verifyFoldRecomputation(
            evals: evals, challenge: challenge, logN: logN, expectedResult: wrongResult)
        expect(!noMatch, "Fold recomputation with wrong data should fail")

        print("  Fold recomputation: \(match ? "PASS" : "FAIL")")
    } catch {
        print("  [FAIL] Fold recomputation: \(error)")
        expect(false, "Fold recomputation threw: \(error)")
    }
}

private func testBatchMerkleVerification() {
    suite("FRI Verifier: Batch Merkle path verification")
    do {
        let verifier = try GPUFRIVerifierEngine()
        let config = GPUFRIConfig(foldingFactor: 2, blowupFactor: 4,
                                   numQueries: 8, finalPolyMaxDegree: 7)
        let proof = try generateProof(logN: 10, config: config, seed: 700)

        // Collect all (authPath, root, leaf) tuples from the proof
        var pathTuples: [(authPath: MerkleAuthPath, root: Fr, leaf: Fr)] = []
        for resp in proof.queryResponses {
            for (layerIdx, layerProof) in resp.layerProofs.enumerated() {
                let root = proof.commitment.layers[layerIdx].merkleTree.root
                pathTuples.append((layerProof.authPath, root, layerProof.evaluation))
            }
        }

        let allValid = verifier.batchVerifyMerklePaths(pathTuples)
        expect(allValid, "Batch Merkle verification should pass for honest proof")

        print("  Batch Merkle: \(allValid ? "PASS" : "FAIL") (\(pathTuples.count) paths)")
    } catch {
        print("  [FAIL] Batch Merkle verification: \(error)")
        expect(false, "Batch Merkle verification threw: \(error)")
    }
}

private func testStructuralValidation() {
    suite("FRI Verifier: Structural validation")
    do {
        let verifier = try GPUFRIVerifierEngine()
        let config = GPUFRIConfig(foldingFactor: 2, blowupFactor: 4,
                                   numQueries: 4, finalPolyMaxDegree: 7)
        let proof = try generateProof(logN: 8, config: config, seed: 800)

        // Valid proof should pass structural validation (implicitly via verify)
        let valid = try verifier.verify(proof: proof)
        expect(valid, "Structurally valid proof should pass")

        print("  Structural validation: \(valid ? "PASS" : "FAIL")")
    } catch {
        print("  [FAIL] Structural validation: \(error)")
        expect(false, "Structural validation threw: \(error)")
    }
}

// MARK: - Public entry point

public func runGPUFRIVerifierTests() {
    testBasicVerification()
    testDetailedVerification()
    testMerklePathVerification()
    testFoldingConsistencyCheck()
    testFinalPolyDegreeCheck()
    testEffectiveDegree()
    testSmallPolynomialRoundTrip()
    testLargerProofRoundTrip()
    testQuickVerification()
    testBatchVerification()
    testDegreeBoundChecking()
    testPolynomialEvaluation()
    testFoldRecomputation()
    testBatchMerkleVerification()
    testStructuralValidation()
}
