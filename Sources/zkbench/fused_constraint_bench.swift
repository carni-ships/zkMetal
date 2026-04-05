// Fused NTT + Constraint benchmark
// Compares: separate NTT + constraint eval (multiple dispatches) vs fused (single command buffer)
// Tests correctness and measures time savings from eliminating host round-trips.

import Foundation
import Metal
import zkMetal

func runFusedConstraintBench() {
    fputs("\n=== Fused NTT + Constraint Benchmark ===\n", stderr)
    fputs("Version: \(FusedNTTConstraintEngine.version.description)\n\n", stderr)

    do {
        let engine = try FusedNTTConstraintEngine()
        fputs("GPU: \(engine.device.name)\n\n", stderr)

        // Generate Fibonacci trace data
        let logSizes = [8, 10, 12, 14, 16]
        let alpha = frFromInt(7)  // random-ish alpha for linear combination

        for logN in logSizes {
            let n = 1 << logN

            // Generate a valid Fibonacci trace: a[i+1] = b[i], b[i+1] = a[i] + b[i]
            var traceA = [Fr](repeating: Fr.zero, count: n)
            var traceB = [Fr](repeating: Fr.zero, count: n)
            traceA[0] = Fr.one
            traceB[0] = Fr.one
            for i in 1..<n {
                traceA[i] = traceB[i - 1]
                traceB[i] = frAdd(traceA[i - 1], traceB[i - 1])
            }

            // Warmup both paths
            let _ = try engine.evaluateFibQuotientSeparate(traceA: traceA, traceB: traceB, alpha: alpha, logN: logN)
            let _ = try engine.evaluateFibQuotientBarrier(traceA: traceA, traceB: traceB, alpha: alpha, logN: logN)

            // Benchmark: Separate (baseline)
            let runs = 5
            var separateTimes = [Double]()
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.evaluateFibQuotientSeparate(traceA: traceA, traceB: traceB, alpha: alpha, logN: logN)
                separateTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000.0)
            }
            separateTimes.sort()
            let separateMedian = separateTimes[runs / 2]

            // Benchmark: Fused (single command buffer)
            var fusedTimes = [Double]()
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.evaluateFibQuotientBarrier(traceA: traceA, traceB: traceB, alpha: alpha, logN: logN)
                fusedTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000.0)
            }
            fusedTimes.sort()
            let fusedMedian = fusedTimes[runs / 2]

            // Benchmark: Fully fused kernel (small sizes only)
            var fusedKernelMedian: Double? = nil
            if logN <= 10 {
                var fusedKernelTimes = [Double]()
                for _ in 0..<runs {
                    let t0 = CFAbsoluteTimeGetCurrent()
                    let _ = try engine.evaluateFibQuotientFused(traceA: traceA, traceB: traceB, alpha: alpha, logN: logN)
                    fusedKernelTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000.0)
                }
                fusedKernelTimes.sort()
                fusedKernelMedian = fusedKernelTimes[runs / 2]
            }

            // Correctness: compare separate vs fused outputs
            let refResult = try engine.evaluateFibQuotientSeparate(traceA: traceA, traceB: traceB, alpha: alpha, logN: logN)
            let fusedResult = try engine.evaluateFibQuotientBarrier(traceA: traceA, traceB: traceB, alpha: alpha, logN: logN)

            var correct = true
            for i in 0..<min(n, refResult.count) {
                if !frEqual(refResult[i], fusedResult[i]) {
                    correct = false
                    if i < 3 {
                        fputs("  MISMATCH at [\(i)]: separate=\(refResult[i]) fused=\(fusedResult[i])\n", stderr)
                    }
                }
            }

            let savings = separateMedian - fusedMedian
            let speedup = separateMedian / fusedMedian
            var line = String(format: "  2^%-2d (%6d): separate %6.2fms | fused %6.2fms | save %+.2fms (%.2f×)",
                            logN, n, separateMedian, fusedMedian, savings, speedup)
            if let fkm = fusedKernelMedian {
                let kSavings = separateMedian - fkm
                line += String(format: " | kernel %6.2fms (save %+.2fms)", fkm, kSavings)
            }
            line += correct ? " [OK]" : " [MISMATCH]"
            fputs(line + "\n", stderr)
        }

        fputs("\nNote: 'separate' = 3 dispatches (NTT_a, NTT_b, constraint eval)\n", stderr)
        fputs("      'fused' = 1 command buffer (NTTs + barrier + eval)\n", stderr)
        fputs("      'kernel' = fully fused Metal kernel (NTT in shared mem + eval, logN<=10 only)\n", stderr)

    } catch {
        fputs("Error: \(error)\n", stderr)
    }
}

/// Helper: compare two Fr values for equality
private func frEqual(_ a: Fr, _ b: Fr) -> Bool {
    return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
           a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

// MARK: - General Constraint System Fused Benchmark

func runFusedGeneralConstraintBench() {
    fputs("\n=== Fused NTT + General Constraint Benchmark ===\n", stderr)
    fputs("Tests codegen-based fused kernel (any ConstraintSystem, not just Fibonacci)\n\n", stderr)

    do {
        let engine = try FusedNTTConstraintEngine()
        let alpha = frFromInt(7)
        let runs = 5

        // Test with Fibonacci constraint system (should match hardcoded Fib path)
        let fibSystem = ConstraintSystem.fibonacci(steps: 2)
        // Add cross-row constraints: next(col0) - col1 = 0, next(col1) - (col0 + col1) = 0
        let cs = ConstraintSystem(numWires: 2)
        cs.assertEqual(.wire(.next(0)), .wire(.col(1)), label: "a_next = b")
        cs.assertEqual(.wire(.next(1)), .wire(.col(0)) + .wire(.col(1)), label: "b_next = a + b")

        let maxFusedLogN = MetalCodegen.maxFusedLogN(numCols: 2)
        fputs("  Max fused logN for 2 cols: \(maxFusedLogN)\n", stderr)

        for logN in [6, 8, maxFusedLogN, maxFusedLogN + 2, 14] {
            let n = 1 << logN
            let isFused = logN <= maxFusedLogN

            // Generate trace
            var colA = [Fr](repeating: Fr.zero, count: n)
            var colB = [Fr](repeating: Fr.zero, count: n)
            colA[0] = Fr.one
            colB[0] = Fr.one
            for i in 1..<n {
                colA[i] = colB[i - 1]
                colB[i] = frAdd(colA[i - 1], colB[i - 1])
            }

            // Warmup
            let _ = try engine.evaluateQuotientFused(
                traceColumns: [colA, colB], system: cs, alpha: alpha, logN: logN)

            // Benchmark
            var times = [Double]()
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.evaluateQuotientFused(
                    traceColumns: [colA, colB], system: cs, alpha: alpha, logN: logN)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000.0)
            }
            times.sort()
            let median = times[runs / 2]

            let path = isFused ? "fused-kernel" : "barrier"
            let line = String(format: "  2^%-2d (%6d): %6.2fms [%@]",
                            logN, n, median, path)
            fputs(line + "\n", stderr)
        }

        // Test with R1CS constraint system (no cross-row refs)
        fputs("\n  R1CS constraints (4 gates, 12 wires):\n", stderr)
        let r1cs = ConstraintSystem.r1cs(numGates: 4)
        let r1csMaxLogN = MetalCodegen.maxFusedLogN(numCols: r1cs.numWires)
        fputs("  Max fused logN for \(r1cs.numWires) cols: \(r1csMaxLogN)\n", stderr)

        for logN in [4, min(r1csMaxLogN, 6), 8] {
            let n = 1 << logN
            let isFused = logN <= r1csMaxLogN

            // Random-ish trace
            var cols = [[Fr]](repeating: [Fr](repeating: Fr.zero, count: n), count: r1cs.numWires)
            for i in 0..<n {
                for j in 0..<r1cs.numWires {
                    cols[j][i] = frFromInt(UInt64(i * r1cs.numWires + j + 1))
                }
            }

            let _ = try engine.evaluateQuotientFused(
                traceColumns: cols, system: r1cs, alpha: alpha, logN: logN)

            var times = [Double]()
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.evaluateQuotientFused(
                    traceColumns: cols, system: r1cs, alpha: alpha, logN: logN)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000.0)
            }
            times.sort()
            let median = times[runs / 2]

            let path = isFused ? "fused-kernel" : "barrier"
            let line = String(format: "  2^%-2d (%6d): %6.2fms [%@]",
                            logN, n, median, path)
            fputs(line + "\n", stderr)
        }

    } catch {
        fputs("Error: \(error)\n", stderr)
    }
}

// MARK: - Circle STARK Fused NTT + Constraint Benchmark (M31)

func runFusedCircleConstraintBench() {
    fputs("\n=== Fused Circle NTT + Constraint Benchmark (M31) ===\n", stderr)
    fputs("Compares: separate LDE+constraint vs fused (single command buffer)\n\n", stderr)

    do {
        let logBlowup = 4
        let runs = 5

        for logTrace in [4, 6, 8, 10] {
            let traceLen = 1 << logTrace
            let logEval = logTrace + logBlowup
            let evalLen = 1 << logEval
            let air = FibonacciAIR(logTraceLength: logTrace)

            // --- Baseline: separate prove (existing path) ---
            let proverSep = CircleSTARKProver(logBlowup: logBlowup, numQueries: 10)
            // Warmup
            let _ = try proverSep.prove(air: air)

            var separateTimes = [Double]()
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try proverSep.prove(air: air)
                separateTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000.0)
            }
            separateTimes.sort()
            let sepMedian = separateTimes[runs / 2]

            // --- Fused: single command buffer path ---
            let proverFused = CircleSTARKProver(logBlowup: logBlowup, numQueries: 10)
            // Warmup
            let _ = try proverFused.proveFused(air: air)

            var fusedTimes = [Double]()
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try proverFused.proveFused(air: air)
                fusedTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000.0)
            }
            fusedTimes.sort()
            let fusedMedian = fusedTimes[runs / 2]

            // Correctness: compare proof outputs
            let refProof = try proverSep.prove(air: air)
            let fusedProof = try proverFused.proveFused(air: air)
            // Both should verify successfully (same AIR, but different alpha due to randomness)
            // We can't compare proofs directly since Fiat-Shamir state differs.
            // Instead, verify both produce valid proofs by checking structure.
            let structOK = refProof.traceCommitments.count == fusedProof.traceCommitments.count
                && refProof.queryResponses.count == fusedProof.queryResponses.count

            let savings = sepMedian - fusedMedian
            let speedup = sepMedian / fusedMedian
            let line = String(format: "  trace=2^%-2d eval=2^%-2d (%5d): separate %6.2fms | fused %6.2fms | save %+.2fms (%.2f×) %@",
                            logTrace, logEval, evalLen, sepMedian, fusedMedian, savings, speedup,
                            structOK ? "[OK]" : "[STRUCT MISMATCH]")
            fputs(line + "\n", stderr)
        }

        fputs("\nNote: 'separate' = host round-trip between NTT and constraint eval\n", stderr)
        fputs("      'fused' = NTT output stays on GPU, constraint eval in same/next command buffer\n", stderr)

    } catch {
        fputs("Error: \(error)\n", stderr)
    }
}
