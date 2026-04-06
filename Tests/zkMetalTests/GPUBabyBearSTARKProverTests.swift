// GPU BabyBear STARK Prover Tests — GPU-accelerated STARK prove pipeline over BabyBear
// Tests: Fibonacci STARK proof, trace commitment, constraint evaluation, proof structure, FRI layer count

import zkMetal
import Foundation

public func runGPUBabyBearSTARKProverTests() {
    suite("GPU BabyBear STARK Prover -- Fibonacci Proof")
    gpuBbStarkTestFibProof()

    suite("GPU BabyBear STARK Prover -- Trace Commitment")
    gpuBbStarkTestTraceCommitment()

    suite("GPU BabyBear STARK Prover -- Constraint Evaluation")
    gpuBbStarkTestConstraintEval()

    suite("GPU BabyBear STARK Prover -- Proof Structure")
    gpuBbStarkTestProofStructure()

    suite("GPU BabyBear STARK Prover -- FRI Layer Count")
    gpuBbStarkTestFRILayerCount()
}

// MARK: - Fibonacci STARK Proof

func gpuBbStarkTestFibProof() {
    do {
        let air = BabyBearFibonacciAIR(logTraceLength: 3)  // 8 rows
        let stark = GPUBabyBearSTARK(config: .fast)
        let (result, verified) = try stark.proveAndVerify(air: air)

        expect(verified, "GPU Fibonacci STARK prove-verify round trip")
        expect(result.totalTimeSeconds > 0, "Total time recorded")
        expect(result.proof.estimatedSizeBytes > 0, "Proof has nonzero size")
        expectEqual(result.traceLength, 8, "Trace length = 8")
        expectEqual(result.numColumns, 2, "Num columns = 2")

        // Prove a larger trace
        let air32 = BabyBearFibonacciAIR(logTraceLength: 5)  // 32 rows
        let result32 = try stark.prove(air: air32)
        let valid32 = try stark.verify(air: air32, proof: result32.proof)
        expect(valid32, "GPU Fibonacci 32-row prove-verify")
        expectEqual(result32.traceLength, 32, "32-row trace length")
    } catch {
        expect(false, "GPU Fibonacci STARK error: \(error)")
    }
}

// MARK: - Trace Commitment

func gpuBbStarkTestTraceCommitment() {
    do {
        let air = BabyBearFibonacciAIR(logTraceLength: 3)
        let stark = GPUBabyBearSTARK(config: .fast)
        let result = try stark.prove(air: air)

        // Trace commitments should be present for each column
        let proof = result.proof.inner
        expectEqual(proof.traceCommitments.count, 2, "2 trace commitments for Fibonacci")

        // Each commitment is 8 Bb elements (Poseidon2 digest)
        for (i, comm) in proof.traceCommitments.enumerated() {
            expectEqual(comm.count, 8, "Trace commitment \(i) has 8 elements")
            // Should not be all zeros
            let nonZero = comm.contains { $0.v != 0 }
            expect(nonZero, "Trace commitment \(i) is non-trivial")
        }

        // Composition commitment should also be 8 elements
        expectEqual(proof.compositionCommitment.count, 8, "Composition commitment has 8 elements")
        let compNonZero = proof.compositionCommitment.contains { $0.v != 0 }
        expect(compNonZero, "Composition commitment is non-trivial")

        // Commitments from different columns should differ
        let comm0 = proof.traceCommitments[0]
        let comm1 = proof.traceCommitments[1]
        let differ = zip(comm0, comm1).contains { $0.v != $1.v }
        expect(differ, "Different columns have different commitments")
    } catch {
        expect(false, "Trace commitment test error: \(error)")
    }
}

// MARK: - Constraint Evaluation

func gpuBbStarkTestConstraintEval() {
    // Verify that the constraint evaluation matches expected behavior
    // by checking the trace satisfies constraints before proving
    let air = BabyBearFibonacciAIR(logTraceLength: 3)
    let trace = air.generateTrace()

    // All transition constraints should be zero on valid trace
    for i in 0..<(air.traceLength - 1) {
        let current = [trace[0][i], trace[1][i]]
        let next = [trace[0][i + 1], trace[1][i + 1]]
        let evals = air.evaluateConstraints(current: current, next: next)
        for (ci, eval) in evals.enumerated() {
            expectEqual(eval.v, UInt32(0),
                       "Constraint \(ci) at row \(i) should be zero")
        }
    }

    // Now prove and verify to ensure GPU constraint evaluation is consistent
    do {
        let stark = GPUBabyBearSTARK(config: .fast)
        let result = try stark.prove(air: air)
        expect(result.constraintEvalTimeSeconds >= 0, "Constraint eval time recorded")

        let valid = try stark.verify(air: air, proof: result.proof)
        expect(valid, "Constraint evaluation produces valid proof")
    } catch {
        expect(false, "Constraint eval test error: \(error)")
    }

    // Test with a generic AIR to ensure constraint evaluation handles different shapes
    do {
        let logN = 3
        let n = 1 << logN
        let air = GenericBabyBearAIR(
            numColumns: 1,
            logTraceLength: logN,
            numConstraints: 1,
            constraintDegree: 1,
            boundaryConstraints: [(column: 0, row: 0, value: Bb.one)],
            traceGenerator: {
                var col = [Bb](repeating: Bb.zero, count: n)
                col[0] = Bb.one
                for i in 1..<n {
                    col[i] = bbAdd(col[i - 1], col[i - 1])
                }
                return [col]
            },
            constraintEvaluator: { current, next in
                let doubled = bbAdd(current[0], current[0])
                return [bbSub(next[0], doubled)]
            }
        )

        let stark = GPUBabyBearSTARK(config: .fast)
        let (_, verified) = try stark.proveAndVerify(air: air)
        expect(verified, "Generic doubling AIR GPU prove-verify")
    } catch {
        expect(false, "Generic AIR constraint eval error: \(error)")
    }
}

// MARK: - Proof Structure

func gpuBbStarkTestProofStructure() {
    do {
        let air = BabyBearFibonacciAIR(logTraceLength: 4)  // 16 rows
        let stark = GPUBabyBearSTARK(config: .fast)
        let result = try stark.prove(air: air)
        let proof = result.proof

        // Check inner proof structure
        let inner = proof.inner
        expectEqual(inner.traceLength, 16, "Inner proof trace length")
        expectEqual(inner.numColumns, 2, "Inner proof num columns")
        expectEqual(inner.logBlowup, 1, "Inner proof log blowup (fast config)")

        // Alpha should be a valid field element (< p)
        expect(inner.alpha.v < Bb.P, "Alpha is valid field element")
        expect(inner.alpha.v != 0, "Alpha is nonzero")

        // Query responses should exist
        expect(inner.queryResponses.count > 0, "Has query responses")
        expect(inner.queryResponses.count <= 20, "Query count <= config numQueries (20)")

        // Each query response should have correct structure
        for (i, qr) in inner.queryResponses.enumerated() {
            expectEqual(qr.traceValues.count, 2,
                       "Query \(i) has 2 trace values")
            expectEqual(qr.traceOpenings.count, 2,
                       "Query \(i) has 2 trace openings")
            expect(qr.queryIndex >= 0, "Query \(i) index non-negative")
        }

        // FRI proof should have rounds
        expect(inner.friProof.rounds.count > 0, "FRI proof has rounds")
        expect(inner.friProof.finalPoly.count > 0, "FRI proof has final polynomial")

        // PoW nonce should be 0 for fast config (grinding disabled)
        expectEqual(proof.powNonce, UInt64(0), "PoW nonce is 0 for fast config")

        // Estimated size should be reasonable
        expect(proof.estimatedSizeBytes > 100, "Proof size > 100 bytes")
        expect(proof.estimatedSizeBytes < 1_000_000, "Proof size < 1MB")

        // Timing breakdown should sum to approximately total
        expect(result.ldeTimeSeconds >= 0, "LDE time non-negative")
        expect(result.commitTimeSeconds >= 0, "Commit time non-negative")
        expect(result.constraintEvalTimeSeconds >= 0, "Constraint eval time non-negative")
        expect(result.friTimeSeconds >= 0, "FRI time non-negative")

        // Summary string should be non-empty
        let summary = result.summary
        expect(summary.count > 0, "Summary non-empty")
        expect(summary.contains("GPU BabyBear STARK"), "Summary contains header")
    } catch {
        expect(false, "Proof structure test error: \(error)")
    }
}

// MARK: - FRI Layer Count

func gpuBbStarkTestFRILayerCount() {
    do {
        // logTrace=3, logBlowup=1 -> logLDE=4, friMaxRemainderLogN=2
        // FRI folds from logN=4 down to 2, so 2 rounds
        let air3 = BabyBearFibonacciAIR(logTraceLength: 3)
        let stark = GPUBabyBearSTARK(config: .fast)
        let result3 = try stark.prove(air: air3)

        let expectedLayers3 = 4 - 2  // logLDE - friMaxRemainderLogN
        expectEqual(result3.proof.friLayerCount, expectedLayers3,
                   "FRI layers for logTrace=3: \(expectedLayers3)")

        // logTrace=5, logBlowup=1 -> logLDE=6, friMaxRemainderLogN=2
        // FRI folds from 6 down to 2, so 4 rounds
        let air5 = BabyBearFibonacciAIR(logTraceLength: 5)
        let result5 = try stark.prove(air: air5)

        let expectedLayers5 = 6 - 2
        expectEqual(result5.proof.friLayerCount, expectedLayers5,
                   "FRI layers for logTrace=5: \(expectedLayers5)")

        // logTrace=4, logBlowup=1 -> logLDE=5, friMaxRemainderLogN=2
        // FRI folds from 5 down to 2, so 3 rounds
        let air4 = BabyBearFibonacciAIR(logTraceLength: 4)
        let result4 = try stark.prove(air: air4)

        let expectedLayers4 = 5 - 2
        expectEqual(result4.proof.friLayerCount, expectedLayers4,
                   "FRI layers for logTrace=4: \(expectedLayers4)")

        // Final polynomial size should match 2^friMaxRemainderLogN
        let fastConfig = BabyBearSTARKConfig.fast
        let expectedFinalSize = 1 << fastConfig.friMaxRemainderLogN
        expectEqual(result3.proof.inner.friProof.finalPoly.count, expectedFinalSize,
                   "Final poly size = 2^friMaxRemainderLogN")
        expectEqual(result5.proof.inner.friProof.finalPoly.count, expectedFinalSize,
                   "Final poly size consistent across trace sizes")

        // Each FRI round should have a non-trivial commitment
        for (ri, round) in result5.proof.inner.friProof.rounds.enumerated() {
            expectEqual(round.commitment.count, 8,
                       "FRI round \(ri) commitment has 8 elements")
            let nonZero = round.commitment.contains { $0.v != 0 }
            expect(nonZero, "FRI round \(ri) commitment is non-trivial")
        }
    } catch {
        expect(false, "FRI layer count test error: \(error)")
    }
}
