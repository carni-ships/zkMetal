// Circle STARK benchmark - Full prove + verify cycle over M31
import zkMetal
import Foundation

public func runCircleSTARKBench() {
    fputs("\n--- Circle STARK Benchmark (Mersenne31) ---\n", stderr)

    // --- Correctness test: Fibonacci AIR ---
    do {
        let air = FibonacciAIR(logTraceLength: 4)
        let trace = air.generateTrace()

        // Verify trace satisfies constraints
        var allZero = true
        for row in 0..<(air.traceLength - 1) {
            let current = (0..<air.numColumns).map { trace[$0][row] }
            let next = (0..<air.numColumns).map { trace[$0][row + 1] }
            let cvals = air.evaluateConstraints(current: current, next: next)
            for cv in cvals { if cv.v != 0 { allZero = false } }
        }
        fputs("  Fibonacci trace (2^4): \(allZero ? "PASS" : "FAIL")\n", stderr)
    }

    // --- Small prove + verify test ---
    do {
        let air = FibonacciAIR(logTraceLength: 6)
        let prover = CircleSTARKProver(logBlowup: 2, numQueries: 10)
        let verifier = CircleSTARKVerifier()

        let t0 = CFAbsoluteTimeGetCurrent()
        let proof = try prover.prove(air: air)
        let proveTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        let t1 = CFAbsoluteTimeGetCurrent()
        let valid = try verifier.verify(air: air, proof: proof)
        let verifyTime = (CFAbsoluteTimeGetCurrent() - t1) * 1000

        fputs("  Prove+Verify (2^6): prove \(String(format: "%.1f", proveTime))ms, " +
              "verify \(String(format: "%.1f", verifyTime))ms, " +
              "\(valid ? "PASS" : "FAIL"), " +
              "proof \(estimateProofSize(proof)) bytes\n", stderr)
    } catch {
        fputs("  Prove+Verify (2^6): FAIL - \(error)\n", stderr)
    }

    // --- Benchmark at various sizes ---
    let sizes = [8, 10, 12, 14]
    fputs("\n  logN  |  Prove (ms)  |  Verify (ms)  |  Proof (bytes)\n", stderr)
    fputs("  ------|-------------|--------------|---------------\n", stderr)

    for logN in sizes {
        do {
            let air = FibonacciAIR(logTraceLength: logN)
            let prover = CircleSTARKProver(logBlowup: 2, numQueries: 16)
            let verifier = CircleSTARKVerifier()

            // Warmup run (primes caches, GPU pipelines, twiddles)
            let _ = try prover.prove(air: air)

            // Timed run with profiling on last size
            prover.profileProve = (logN == sizes.last!)
            var bestProve = Double.infinity
            var proof: CircleSTARKProof!
            for _ in 0..<3 {
                let t0 = CFAbsoluteTimeGetCurrent()
                proof = try prover.prove(air: air)
                let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                if ms < bestProve { bestProve = ms }
            }
            let proveTime = bestProve

            let t1 = CFAbsoluteTimeGetCurrent()
            let valid = try verifier.verify(air: air, proof: proof)
            let verifyTime = (CFAbsoluteTimeGetCurrent() - t1) * 1000

            if !valid {
                fputs("    \(logN)  |  FAIL (verification failed)\n", stderr)
                continue
            }

            fputs("    \(logN)  | \(String(format: "%10.1f", proveTime))  | " +
                  "\(String(format: "%11.1f", verifyTime))  | " +
                  "\(String(format: "%13d", estimateProofSize(proof)))\n", stderr)
        } catch {
            fputs("    \(logN)  |  FAIL: \(error)\n", stderr)
        }
    }

    // --- Fused vs Separate comparison ---
    fputs("\n  Fused NTT+Constraint comparison:\n", stderr)
    for logN in [8, 10, 12, 14] {
        do {
            let air = FibonacciAIR(logTraceLength: logN)

            // Separate (baseline)
            let proverSep = CircleSTARKProver(logBlowup: 2, numQueries: 16)
            let _ = try proverSep.prove(air: air) // warmup
            let t0 = CFAbsoluteTimeGetCurrent()
            let _ = try proverSep.prove(air: air)
            let sepMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

            // Fused
            let proverFused = CircleSTARKProver(logBlowup: 2, numQueries: 16)
            let _ = try proverFused.proveFused(air: air) // warmup
            let t1 = CFAbsoluteTimeGetCurrent()
            let _ = try proverFused.proveFused(air: air)
            let fusedMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000

            let speedup = sepMs / fusedMs
            fputs(String(format: "    2^%-2d: separate %6.1fms | fused %6.1fms | %.2f×\n",
                        logN, sepMs, fusedMs, speedup), stderr)
        } catch {
            fputs("    2^\(logN): ERROR - \(error)\n", stderr)
        }
    }

    // --- Soundness test ---
    do {
        let air = FibonacciAIR(logTraceLength: 6)
        let prover = CircleSTARKProver(logBlowup: 2, numQueries: 10)
        let verifier = CircleSTARKVerifier()

        let proof = try prover.prove(air: air)
        let valid = try verifier.verify(air: air, proof: proof)
        fputs("  Soundness: valid proof verifies: \(valid ? "PASS" : "FAIL")\n", stderr)

        // Tamper with trace commitment
        var badCommitments = proof.traceCommitments
        if !badCommitments.isEmpty && badCommitments[0].count > 0 {
            badCommitments[0][0] ^= 0xFF
        }
        let tampered = CircleSTARKProof(
            traceCommitments: badCommitments,
            compositionCommitment: proof.compositionCommitment,
            friProof: proof.friProof,
            queryResponses: proof.queryResponses,
            alpha: proof.alpha,
            traceLength: proof.traceLength,
            numColumns: proof.numColumns,
            logBlowup: proof.logBlowup
        )
        do {
            let _ = try verifier.verify(air: air, proof: tampered)
            fputs("  Soundness: tampered proof rejected: FAIL (accepted!)\n", stderr)
        } catch {
            fputs("  Soundness: tampered proof rejected: PASS\n", stderr)
        }
    } catch {
        fputs("  Soundness test error: \(error)\n", stderr)
    }
}

/// Estimate proof size in bytes
func estimateProofSize(_ proof: CircleSTARKProof) -> Int {
    var size = 0
    size += proof.traceCommitments.count * 32
    size += 32  // composition commitment
    for round in proof.friProof.rounds {
        size += 32  // commitment
        for (_, _, path) in round.queryResponses {
            size += 8  // two M31 values
            size += path.count * 32
        }
    }
    size += 4  // final value
    for qr in proof.queryResponses {
        size += qr.traceValues.count * 4
        for path in qr.tracePaths { size += path.count * 32 }
        size += 4  // composition value
        size += qr.compositionPath.count * 32
    }
    return size
}

// MARK: - Fused STARK Round Kernel Benchmark

/// Benchmark the fused constraint eval + FRI fold kernels.
/// This measures the performance improvement from fusing the constraint evaluation
/// with the first FRI fold in a single GPU dispatch.
///
/// Fused kernels:
/// - circle_fib_constraint_fold_first: constraint eval + 1st FRI fold, outputs n/2
/// - circle_fib_constraint_fold_2r: constraint eval + 2 FRI folds, outputs n/4
public func runCircleSTARKFusedRoundBench() {
    fputs("\n--- Circle STARK Fused Round Kernel Benchmark ---\n", stderr)
    fputs("Measures: constraint eval + FRI fold (separate vs fused)\n\n", stderr)

    for logN in [8, 10, 12, 14, 16] {
        do {
            let air = FibonacciAIR(logTraceLength: logN)
            let prover = CircleSTARKProver(logBlowup: 2, numQueries: 16)

            // Warmup
            let _ = try prover.generateTraceGPU(air: air)

            let metrics = try prover.benchmarkFusedConstraintFold(air: air)

            if metrics.fusedAvailable {
                let speedup = metrics.speedup ?? 1.0
                let fold2rStr = metrics.fold2rMs.map { String(format: " (2r: %.1fms)", $0) } ?? ""
                fputs(String(format: "  2^%-2d: separate cst+fold %6.1fms | " +
                            "fused %6.1fms | %.2fx speedup%@\n",
                            logN, metrics.constraintMs + metrics.foldMs,
                            metrics.fusedMs ?? 0, speedup, fold2rStr), stderr)
            } else {
                fputs(String(format: "  2^%-2d: fused kernel unavailable (XPC compilation issue)\n", logN), stderr)
            }
        } catch {
            fputs("  2^\(logN): ERROR - \(error)\n", stderr)
        }
    }

    fputs("\n  Legend:\n", stderr)
    fputs("    separate: constraint eval + FRI fold in TWO GPU dispatches\n", stderr)
    fputs("    fused:    constraint eval + FRI fold in ONE GPU dispatch\n", stderr)
    fputs("    2r:       constraint eval + 2 FRI folds in ONE dispatch (outputs n/4)\n\n", stderr)

    // Full prover comparison
    fputs("  Full prover comparison (prove vs proveFused):\n", stderr)
    for logN in [8, 10, 12] {
        do {
            let air = FibonacciAIR(logTraceLength: logN)

            // Standard prover
            let proverSep = CircleSTARKProver(logBlowup: 2, numQueries: 16)
            let _ = try proverSep.prove(air: air)  // warmup
            let t0 = CFAbsoluteTimeGetCurrent()
            let _ = try proverSep.prove(air: air)
            let sepMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000

            // Fused prover
            let proverFused = CircleSTARKProver(logBlowup: 2, numQueries: 16)
            let _ = try proverFused.proveFused(air: air)  // warmup
            let t1 = CFAbsoluteTimeGetCurrent()
            let _ = try proverFused.proveFused(air: air)
            let fusedMs = (CFAbsoluteTimeGetCurrent() - t1) * 1000

            let speedup = sepMs / fusedMs
            fputs(String(format: "    2^%-2d: standard %6.1fms | fused %6.1fms | %.2fx\n",
                        logN, sepMs, fusedMs, speedup), stderr)
        } catch {
            fputs("    2^\(logN): ERROR - \(error)\n", stderr)
        }
    }
}
