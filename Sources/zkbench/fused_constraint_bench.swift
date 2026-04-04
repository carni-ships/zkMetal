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
