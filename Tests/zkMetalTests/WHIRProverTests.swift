// WHIR Prover/Verifier Tests
//
// Tests for the complete WHIR polynomial IOP:
//   1. Prove + verify degree-2^10 polynomial
//   2. Prove + verify degree-2^14 polynomial
//   3. Tampered proof rejected (wrong query response)
//   4. Wrong commitment rejected
//   5. Proof size scales logarithmically with degree
//   6. Configurable parameters

import Foundation
import zkMetal

public func runWHIRProverTests() {
    suite("WHIR Prover/Verifier IOP")

    // Helper: generate polynomial evaluations f(omega^i) for i in 0..<n
    // Uses a simple polynomial f(x) = sum_{j=0}^{deg} c_j * x^j
    // where c_j are deterministic field elements.
    func generateEvaluations(logN: Int) -> [Fr] {
        let n = 1 << logN
        var evals = [Fr](repeating: Fr.zero, count: n)
        // Evaluate polynomial with coefficients c_j = j + 1 at domain points
        let omega = frRootOfUnity(logN: logN)
        for i in 0..<n {
            // Simple: f(omega^i) = i + 1 (not a real degree-bound polynomial,
            // but sufficient for testing the IOP mechanics)
            evals[i] = frFromInt(UInt64(i + 1))
        }
        return evals
    }

    // ------------------------------------------------------------------
    // 1. Prove and verify degree-2^10 polynomial
    // ------------------------------------------------------------------
    do {
        let config = WHIRIOPConfig.fast
        let prover = try! WHIRIOPProver(config: config)
        let verifier = WHIRIOPVerifier()

        let logN = 10
        let evals = generateEvaluations(logN: logN)
        let proof = try! prover.prove(evaluations: evals)

        // Check proof structure
        expect(proof.domainSize == 1 << logN,
               "2^10: domainSize = \(1 << logN)")
        expect(proof.numRounds > 0,
               "2^10: at least one folding round")
        expect(!proof.initialCommitment.isZero,
               "2^10: initial commitment is non-zero")
        expect(!proof.finalPolynomial.isEmpty,
               "2^10: final polynomial is non-empty")

        // Verify
        let ok = verifier.verify(proof: proof)
        expect(ok, "2^10: proof verifies correctly")
        print("  2^10 prove+verify: OK (rounds=\(proof.numRounds), proof=\(proof.proofSizeBytes) bytes)")
    }

    // ------------------------------------------------------------------
    // 2. Prove and verify degree-2^14 polynomial
    // ------------------------------------------------------------------
    do {
        let config = WHIRIOPConfig.fast
        let prover = try! WHIRIOPProver(config: config)
        let verifier = WHIRIOPVerifier()

        let logN = 14
        let evals = generateEvaluations(logN: logN)
        let proof = try! prover.prove(evaluations: evals)

        expect(proof.domainSize == 1 << logN,
               "2^14: domainSize = \(1 << logN)")

        let ok = verifier.verify(proof: proof)
        expect(ok, "2^14: proof verifies correctly")
        print("  2^14 prove+verify: OK (rounds=\(proof.numRounds), proof=\(proof.proofSizeBytes) bytes)")
    }

    // ------------------------------------------------------------------
    // 3. Tampered proof rejected (wrong query response)
    // ------------------------------------------------------------------
    do {
        let config = WHIRIOPConfig.fast
        let prover = try! WHIRIOPProver(config: config)
        let verifier = WHIRIOPVerifier()

        let logN = 10
        let evals = generateEvaluations(logN: logN)
        let proof = try! prover.prove(evaluations: evals)

        // Sanity: original proof verifies
        expect(verifier.verify(proof: proof), "tamper test: original proof valid")

        // Tamper with a query response value
        var tamperedResponses = proof.queryResponses
        if !tamperedResponses.isEmpty && !tamperedResponses[0].isEmpty {
            let qr = tamperedResponses[0][0]
            var badValues = qr.values
            // Flip one evaluation value
            badValues[0] = frAdd(badValues[0], Fr.one)
            tamperedResponses[0][0] = WHIRQueryResponse(
                foldedIndex: qr.foldedIndex,
                values: badValues,
                merklePaths: qr.merklePaths)
        }

        let tamperedProof = WHIRIOPProof(
            initialCommitment: proof.initialCommitment,
            roundCommitments: proof.roundCommitments,
            challenges: proof.challenges,
            weightSeeds: proof.weightSeeds,
            queryResponses: tamperedResponses,
            weightedSums: proof.weightedSums,
            hashCommitments: proof.hashCommitments,
            finalPolynomial: proof.finalPolynomial,
            config: proof.config,
            domainSize: proof.domainSize
        )

        let bad = verifier.verify(proof: tamperedProof)
        expect(!bad, "tamper test: modified query response rejected")
        print("  Tampered query response: correctly rejected")
    }

    // ------------------------------------------------------------------
    // 4. Wrong commitment rejected
    // ------------------------------------------------------------------
    do {
        let config = WHIRIOPConfig.fast
        let prover = try! WHIRIOPProver(config: config)
        let verifier = WHIRIOPVerifier()

        let logN = 10
        let evals = generateEvaluations(logN: logN)
        let proof = try! prover.prove(evaluations: evals)

        // Replace initial commitment with a random value
        let fakeRoot = frFromInt(999999)
        let badProof = WHIRIOPProof(
            initialCommitment: fakeRoot,
            roundCommitments: proof.roundCommitments,
            challenges: proof.challenges,
            weightSeeds: proof.weightSeeds,
            queryResponses: proof.queryResponses,
            weightedSums: proof.weightedSums,
            hashCommitments: proof.hashCommitments,
            finalPolynomial: proof.finalPolynomial,
            config: proof.config,
            domainSize: proof.domainSize
        )

        let bad = verifier.verify(proof: badProof)
        expect(!bad, "wrong commitment: proof rejected")
        print("  Wrong commitment: correctly rejected")
    }

    // ------------------------------------------------------------------
    // 5. Proof size scales logarithmically with degree
    // ------------------------------------------------------------------
    do {
        let config = WHIRIOPConfig.fast
        let prover = try! WHIRIOPProver(config: config)

        var sizes: [(Int, Int)] = []
        for logN in [8, 10, 12, 14] {
            let evals = generateEvaluations(logN: logN)
            let proof = try! prover.prove(evaluations: evals)
            sizes.append((logN, proof.proofSizeBytes))
        }

        // Check that doubling the degree does not double the proof size.
        // For logarithmic scaling, going from 2^8 to 2^14 (64x domain)
        // should yield much less than 64x proof size increase.
        let ratio = Double(sizes[3].1) / Double(sizes[0].1)
        expect(ratio < 16.0,
               "logarithmic scaling: 2^14/2^8 proof size ratio = \(String(format: "%.1f", ratio)) < 16")

        // Also check rounds grow logarithmically
        for (logN, size) in sizes {
            let evals = generateEvaluations(logN: logN)
            let proof = try! prover.prove(evaluations: evals)
            let expectedMaxRounds = logN  // at most logN rounds (folding by 2 each time)
            expect(proof.numRounds <= expectedMaxRounds,
                   "logN=\(logN): rounds=\(proof.numRounds) <= \(expectedMaxRounds)")
        }

        let sizeStrs = sizes.map { "2^\($0.0)=\($0.1)B" }.joined(separator: ", ")
        print("  Proof sizes: \(sizeStrs) (ratio=\(String(format: "%.1f", ratio))x)")
    }

    // ------------------------------------------------------------------
    // 6. Different configurations produce valid proofs
    // ------------------------------------------------------------------
    do {
        let logN = 10
        let evals = generateEvaluations(logN: logN)
        let verifier = WHIRIOPVerifier()

        // Config with foldingFactor=2 (more rounds, smaller fold)
        let config2 = WHIRIOPConfig(numQueries: 4, foldingFactor: 2,
                                     securityLevel: 40, finalPolyMaxSize: 8)
        let prover2 = try! WHIRIOPProver(config: config2)
        let proof2 = try! prover2.prove(evaluations: evals)
        let ok2 = verifier.verify(proof: proof2)
        expect(ok2, "foldingFactor=2: proof verifies")

        // Config with foldingFactor=8 (fewer rounds, larger fold)
        let config8 = WHIRIOPConfig(numQueries: 4, foldingFactor: 8,
                                     securityLevel: 40, finalPolyMaxSize: 16)
        let prover8 = try! WHIRIOPProver(config: config8)
        let proof8 = try! prover8.prove(evaluations: evals)
        let ok8 = verifier.verify(proof: proof8)
        expect(ok8, "foldingFactor=8: proof verifies")

        // More rounds with foldingFactor=2 than foldingFactor=8
        expect(proof2.numRounds > proof8.numRounds,
               "foldingFactor=2 has more rounds than foldingFactor=8")

        print("  Config variants: fold=2 rounds=\(proof2.numRounds), fold=8 rounds=\(proof8.numRounds)")
    }
}
