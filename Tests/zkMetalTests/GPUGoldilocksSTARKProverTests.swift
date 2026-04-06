// GPU Goldilocks STARK Prover Tests — end-to-end prove + verify over Goldilocks field
// Tests: Fibonacci STARK proof, trace dimensions, constraint eval, proof non-empty, verify accepts

import Foundation
import zkMetal

public func runGPUGoldilocksSTARKProverTests() {
    suite("GPU Goldilocks STARK Prover -- Fibonacci STARK proof")
    gpuGlStarkTestFibProveVerify()

    suite("GPU Goldilocks STARK Prover -- Trace dimensions")
    gpuGlStarkTestTraceDimensions()

    suite("GPU Goldilocks STARK Prover -- Constraint evaluation")
    gpuGlStarkTestConstraintEval()

    suite("GPU Goldilocks STARK Prover -- Proof non-empty")
    gpuGlStarkTestProofNonEmpty()

    suite("GPU Goldilocks STARK Prover -- Verify accepts valid proof")
    gpuGlStarkTestVerifyAccepts()

    suite("GPU Goldilocks STARK Prover -- Deep composition")
    gpuGlStarkTestDeepComposition()

    suite("GPU Goldilocks STARK Prover -- Benchmark")
    gpuGlStarkBenchmark()
}

// MARK: - Fibonacci STARK proof

func gpuGlStarkTestFibProveVerify() {
    do {
        let air = GoldilocksFibonacciAIR(logTraceLength: 3)  // 8 rows
        let prover = GPUGoldilocksSTARKProver(config: .fast)
        let (result, verified) = try prover.proveAndVerify(air: air)

        expect(verified, "Fibonacci prove-verify round trip succeeds")
        expectEqual(result.proof.traceLength, 8, "Proof trace length = 8")
        expectEqual(result.proof.numColumns, 2, "Proof num columns = 2")
        expect(result.proof.queryResponses.count > 0, "Has query responses")
        expect(result.proof.friProof.rounds.count > 0, "Has FRI rounds")
        expect(result.proveTimeSeconds > 0, "Prove time recorded")
        expect(result.proofSizeBytes > 0, "Proof has nonzero size")
    } catch {
        expect(false, "Fibonacci STARK prove-verify failed: \(error)")
    }
}

// MARK: - Trace dimensions

func gpuGlStarkTestTraceDimensions() {
    // logN=3 -> 8 rows, 2 columns
    let air3 = GoldilocksFibonacciAIR(logTraceLength: 3)
    let trace3 = air3.generateTrace()
    expectEqual(trace3.count, 2, "Fib AIR has 2 columns")
    expectEqual(trace3[0].count, 8, "logN=3 -> 8 rows")
    expectEqual(trace3[1].count, 8, "Column B also 8 rows")

    // logN=4 -> 16 rows
    let air4 = GoldilocksFibonacciAIR(logTraceLength: 4)
    let trace4 = air4.generateTrace()
    expectEqual(trace4[0].count, 16, "logN=4 -> 16 rows")

    // logN=5 -> 32 rows
    let air5 = GoldilocksFibonacciAIR(logTraceLength: 5)
    let trace5 = air5.generateTrace()
    expectEqual(trace5[0].count, 32, "logN=5 -> 32 rows")

    // Verify Fibonacci values
    expectEqual(trace3[0][0].v, 1, "a[0] = 1")
    expectEqual(trace3[1][0].v, 1, "b[0] = 1")
    expectEqual(trace3[0][1].v, 1, "a[1] = b[0] = 1")
    expectEqual(trace3[1][1].v, 2, "b[1] = a[0]+b[0] = 2")
    expectEqual(trace3[0][3].v, 3, "a[3] = b[2] = 3")
    expectEqual(trace3[1][3].v, 5, "b[3] = a[2]+b[2] = 5")

    // Custom initial values
    let airCustom = GoldilocksFibonacciAIR(logTraceLength: 2, a0: Gl(v: 10), b0: Gl(v: 20))
    let traceCustom = airCustom.generateTrace()
    expectEqual(traceCustom[0][0].v, 10, "Custom a[0] = 10")
    expectEqual(traceCustom[1][0].v, 20, "Custom b[0] = 20")
    expectEqual(traceCustom[0][1].v, 20, "Custom a[1] = 20")
    expectEqual(traceCustom[1][1].v, 30, "Custom b[1] = 30")
}

// MARK: - Constraint evaluation

func gpuGlStarkTestConstraintEval() {
    let air = GoldilocksFibonacciAIR(logTraceLength: 3)
    let trace = air.generateTrace()

    // All transition constraints should evaluate to zero on a valid trace
    for i in 0..<(air.traceLength - 1) {
        let current = [trace[0][i], trace[1][i]]
        let next = [trace[0][i + 1], trace[1][i + 1]]
        let evals = air.evaluateConstraints(current: current, next: next)
        expectEqual(evals.count, 2, "Fibonacci has 2 constraints")
        expect(evals[0].v == 0, "C0 zero at row \(i)")
        expect(evals[1].v == 0, "C1 zero at row \(i)")
    }

    // Intentionally wrong next values should produce non-zero constraints
    let badCurrent = [Gl(v: 1), Gl(v: 1)]
    let badNext = [Gl(v: 999), Gl(v: 999)]
    let badEvals = air.evaluateConstraints(current: badCurrent, next: badNext)
    expect(badEvals[0].v != 0 || badEvals[1].v != 0,
           "Bad transition produces non-zero constraint")

    // verifyTrace should pass for valid trace
    let err = air.verifyTrace(trace)
    expect(err == nil, "Valid trace passes verifyTrace: \(err ?? "")")

    // verifyTrace should reject tampered trace
    var badTrace = trace
    badTrace[0][2] = Gl(v: 999)
    let err2 = air.verifyTrace(badTrace)
    expect(err2 != nil, "Tampered trace rejected by verifyTrace")
}

// MARK: - Proof non-empty

func gpuGlStarkTestProofNonEmpty() {
    do {
        let air = GoldilocksFibonacciAIR(logTraceLength: 3)
        let prover = GPUGoldilocksSTARKProver(config: .fast)
        let result = try prover.prove(air: air)
        let proof = result.proof

        // Structural checks
        expect(proof.traceCommitments.count == 2, "2 trace commitments")
        for (i, commit) in proof.traceCommitments.enumerated() {
            expectEqual(commit.count, 4, "Trace commitment \(i) is 4-element digest")
            expect(commit.contains { $0.v != 0 }, "Trace commitment \(i) non-trivial")
        }
        expectEqual(proof.compositionCommitment.count, 4, "Composition commitment is 4-element")
        expect(proof.compositionCommitment.contains { $0.v != 0 },
               "Composition commitment non-trivial")

        // FRI proof non-empty
        expect(proof.friProof.rounds.count > 0, "FRI has rounds")
        expect(proof.friProof.finalPoly.count > 0, "FRI has final polynomial")
        expect(proof.friProof.queryIndices.count > 0, "FRI has query indices")

        // Query responses
        expect(proof.queryResponses.count > 0, "Has query responses")
        for (i, qr) in proof.queryResponses.enumerated() {
            expectEqual(qr.traceValues.count, 2, "Query \(i) has 2 trace values")
            expectEqual(qr.traceOpenings.count, 2, "Query \(i) has 2 trace openings")
            expect(qr.compositionOpening.path.count > 0,
                   "Query \(i) composition opening has path")
        }

        // Proof size
        expect(proof.estimatedSizeBytes > 0, "Proof size positive")
        print("  Proof size: \(proof.estimatedSizeBytes) bytes")
        print("  FRI rounds: \(proof.friProof.rounds.count)")
        print("  Queries: \(proof.queryResponses.count)")
    } catch {
        expect(false, "Proof generation failed: \(error)")
    }
}

// MARK: - Verify accepts valid proof

func gpuGlStarkTestVerifyAccepts() {
    do {
        // Small trace
        let air8 = GoldilocksFibonacciAIR(logTraceLength: 3)
        let prover = GPUGoldilocksSTARKProver(config: .fast)

        let result8 = try prover.prove(air: air8)
        let valid8 = try prover.verify(air: air8, proof: result8.proof)
        expect(valid8, "Verifier accepts valid 8-row proof")

        // Larger trace
        let air32 = GoldilocksFibonacciAIR(logTraceLength: 5)
        let result32 = try prover.prove(air: air32)
        let valid32 = try prover.verify(air: air32, proof: result32.proof)
        expect(valid32, "Verifier accepts valid 32-row proof")

        // Custom initial values
        let airCustom = GoldilocksFibonacciAIR(logTraceLength: 3, a0: Gl(v: 5), b0: Gl(v: 8))
        let resultCustom = try prover.prove(air: airCustom)
        let validCustom = try prover.verify(air: airCustom, proof: resultCustom.proof)
        expect(validCustom, "Verifier accepts custom-initial proof")

        // Tampered proof should be rejected
        let tamperedProof = GoldilocksSTARKProof(
            traceCommitments: result8.proof.traceCommitments,
            compositionCommitment: result8.proof.compositionCommitment,
            friProof: result8.proof.friProof,
            queryResponses: result8.proof.queryResponses,
            alpha: Gl(v: 12345),  // wrong alpha
            traceLength: result8.proof.traceLength,
            numColumns: result8.proof.numColumns,
            logBlowup: result8.proof.logBlowup
        )
        var rejected = false
        do {
            _ = try prover.verify(air: air8, proof: tamperedProof)
        } catch {
            rejected = true
        }
        expect(rejected, "Tampered proof (wrong alpha) rejected")
    } catch {
        expect(false, "Verify test failed: \(error)")
    }
}

// MARK: - Deep composition

func gpuGlStarkTestDeepComposition() {
    do {
        let air = GoldilocksFibonacciAIR(logTraceLength: 3)
        let prover = GPUGoldilocksSTARKProver(config: .fast)
        let result = try prover.prove(air: air)

        // Deep composition should be populated
        let dc = result.deepComposition
        expectEqual(dc.traceAtZ.count, 2, "Deep comp has 2 trace evaluations at z")
        expect(dc.z.v != 0, "OOD point z is non-zero")
        expect(dc.digest.count > 0, "Deep comp digest is non-empty")
        expect(dc.digest.contains { $0.v != 0 }, "Deep comp digest is non-trivial")

        // Summary string
        let summary = result.summary
        expect(summary.contains("GPU Goldilocks STARK"), "Summary has correct header")
        expect(summary.count > 0, "Summary non-empty")
    } catch {
        expect(false, "Deep composition test failed: \(error)")
    }
}

// MARK: - Benchmark

func gpuGlStarkBenchmark() {
    do {
        let air = GoldilocksFibonacciAIR(logTraceLength: 5)  // 32 rows
        let prover = GPUGoldilocksSTARKProver(config: .fast)

        // Warm up
        _ = try prover.prove(air: air)

        // Timed run
        let t0 = CFAbsoluteTimeGetCurrent()
        let result = try prover.prove(air: air)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0

        print(String(format: "  GPU Goldilocks STARK Fibonacci (32 rows): %.3fms, proof %d bytes, GPU: %@",
                      elapsed * 1000, result.proofSizeBytes, result.usedGPU ? "yes" : "no"))
        expect(elapsed < 30.0, "Prove completes in < 30s")
    } catch {
        expect(false, "Benchmark error: \(error)")
    }
}
