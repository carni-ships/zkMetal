// WHIR Benchmark — proximity testing performance and correctness vs FRI
import zkMetal
import Foundation

public func runWHIRBench() {
    fputs("=== WHIR Benchmark ===\n", stderr)

    do {
        // MARK: - Correctness Tests

        fputs("\n--- Correctness verification ---\n", stderr)

        // Generate test evaluations
        var rng: UInt64 = 0xDEAD_BEEF_1234
        func nextRng() -> UInt64 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return rng >> 32
        }

        let testLogN = 10
        let testN = 1 << testLogN
        var testEvals = [Fr](repeating: Fr.zero, count: testN)
        for i in 0..<testN {
            testEvals[i] = frFromInt(nextRng())
        }

        // Test commit
        let engine = try WHIREngine(numQueries: 16, reductionFactor: 4)
        let commitment = try engine.commit(evaluations: testEvals)
        print("  Commit 2^\(testLogN): root computed, tree size = \(commitment.tree.count)")

        // Test prove
        let proof = try engine.prove(evaluations: testEvals)
        print("  Prove 2^\(testLogN): \(proof.numRounds) rounds, final poly size = \(proof.finalPoly.count)")
        print("  Proof size: \(proof.proofSizeBytes) bytes (\(String(format: "%.1f", Double(proof.proofSizeBytes) / 1024)) KB)")

        // Test verify (full, with original evaluations)
        let verified = engine.verifyFull(proof: proof, evaluations: testEvals)
        print("  Verify (full): \(verified ? "PASS" : "FAIL")")

        // Test verify (without original evaluations — transcript-only)
        let verifiedLight = engine.verify(proof: proof, evaluations: testEvals)
        print("  Verify (light): \(verifiedLight ? "PASS" : "FAIL")")

        // Test with different parameters
        let engine8 = try WHIREngine(numQueries: 8, reductionFactor: 2)
        let proof8 = try engine8.prove(evaluations: testEvals)
        let verified8 = engine8.verifyFull(proof: proof8, evaluations: testEvals)
        print("  Verify (queries=8, reduce=2): \(verified8 ? "PASS" : "FAIL"), \(proof8.numRounds) rounds, \(proof8.proofSizeBytes) bytes")

        // MARK: - Performance Benchmarks

        print("\n--- WHIR vs FRI Performance ---")
        let benchSizes = [14]
        let friEngine = try FRIEngine()

        for logN in benchSizes {
            let n = 1 << logN
            var evals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                evals[i] = frFromInt(nextRng())
            }

            // WHIR prove
            let whirEngine = try WHIREngine(numQueries: 32, reductionFactor: 4)

            // Warmup
            let _ = try whirEngine.prove(evaluations: evals)

            var whirTimes = [Double]()
            var whirProofSize = 0
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let p = try whirEngine.prove(evaluations: evals)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                whirTimes.append(elapsed)
                whirProofSize = p.proofSizeBytes
            }
            whirTimes.sort()
            let whirMedian = whirTimes[2]

            // WHIR verify
            let whirProof = try whirEngine.prove(evaluations: evals)
            var whirVerifyTimes = [Double]()
            for _ in 0..<10 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = whirEngine.verifyFull(proof: whirProof, evaluations: evals)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                whirVerifyTimes.append(elapsed)
            }
            whirVerifyTimes.sort()
            let whirVerifyMedian = whirVerifyTimes[5]

            // FRI prove (commit phase)
            var betas = [Fr]()
            for i in 0..<logN {
                betas.append(frFromInt(UInt64(i + 1) * 17))
            }

            // Warmup
            let _ = try friEngine.commitPhase(evals: evals, betas: betas)

            var friTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try friEngine.commitPhase(evals: evals, betas: betas)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                friTimes.append(elapsed)
            }
            friTimes.sort()
            let friMedian = friTimes[2]

            // FRI proof size estimate: layers + roots + final value
            let friCommitment = try friEngine.commitPhase(evals: evals, betas: betas)
            let frSize = MemoryLayout<Fr>.stride
            var friProofSize = friCommitment.roots.count * frSize  // roots
            friProofSize += frSize  // final value
            // Query proofs: 2 queries * (2 evals + logN path) per layer
            let numFRIQueries = 2
            for layer in friCommitment.layers {
                let layerLogN = Int(log2(Double(layer.count)))
                friProofSize += numFRIQueries * (2 * frSize + layerLogN * frSize)
            }

            // FRI verify (2 queries to keep Merkle rebuild reasonable)
            let queryIndices: [UInt32] = [0, 42]
            let friQueries = try friEngine.queryPhase(commitment: friCommitment, queryIndices: queryIndices)
            var friVerifyTimes = [Double]()
            for _ in 0..<10 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = friEngine.verify(commitment: friCommitment, queries: friQueries)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                friVerifyTimes.append(elapsed)
            }
            friVerifyTimes.sort()
            let friVerifyMedian = friVerifyTimes[5]

            print(String(format: "  2^%-2d = %7d elements:", logN, n))
            print(String(format: "    WHIR  prover: %7.2fms | verify: %7.3fms | proof: %6d bytes (%.1f KB)",
                        whirMedian, whirVerifyMedian, whirProofSize, Double(whirProofSize) / 1024))
            print(String(format: "    FRI   prover: %7.2fms | verify: %7.3fms | proof: %6d bytes (%.1f KB)",
                        friMedian, friVerifyMedian, friProofSize, Double(friProofSize) / 1024))

            let proverRatio = whirMedian / friMedian
            let proofRatio = Double(whirProofSize) / Double(friProofSize)
            print(String(format: "    Ratio: prover %.2fx | proof size %.2fx %s",
                        proverRatio, proofRatio,
                        proofRatio < 1.0 ? "(WHIR smaller)" : "(FRI smaller)"))
        }

        // MARK: - Parameter Sensitivity

        print("\n--- WHIR Parameter Sensitivity (2^14) ---")
        let paramN = 1 << 14
        var paramEvals = [Fr](repeating: Fr.zero, count: paramN)
        for i in 0..<paramN {
            paramEvals[i] = frFromInt(nextRng())
        }

        let configs: [(queries: Int, reduction: Int)] = [
            (8, 2), (16, 2), (32, 2),
            (8, 4), (16, 4), (32, 4),
            (16, 8), (32, 8),
        ]

        print(String(format: "  %-10s %-10s %-10s %-10s %-10s", "Queries", "Reduce", "Prove(ms)", "Verify(ms)", "Proof(KB)"))
        for config in configs {
            let eng = try WHIREngine(numQueries: config.queries, reductionFactor: config.reduction)

            // Warmup
            let _ = try eng.prove(evaluations: paramEvals)

            var times = [Double]()
            var proofSize = 0
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let p = try eng.prove(evaluations: paramEvals)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
                proofSize = p.proofSizeBytes
            }
            times.sort()

            let p = try eng.prove(evaluations: paramEvals)
            var vtimes = [Double]()
            for _ in 0..<10 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = eng.verifyFull(proof: p, evaluations: paramEvals)
                vtimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            vtimes.sort()

            print(String(format: "  %-10d %-10d %-10.2f %-10.3f %-10.1f",
                        config.queries, config.reduction, times[2], vtimes[5],
                        Double(proofSize) / 1024))
        }

    } catch {
        print("Error: \(error)")
    }
}
