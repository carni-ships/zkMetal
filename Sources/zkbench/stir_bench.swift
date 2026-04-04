// STIR Benchmark — proximity testing performance and correctness
// Compares STIR vs FRI: prover time, queries needed, proof size
import zkMetal
import Foundation

public func runSTIRBench() {
    print("=== STIR Benchmark (Shift To Improve Rate) ===")

    do {
        let engine = try STIREngine()
        let friEngine = engine.friEngine
        var rng: UInt64 = 0xDEAD_BEEF_CAFE_0517

        // --- Correctness: domain shift ---
        print("\n--- Domain shift correctness ---")
        let shiftLogN = 8
        let shiftN = 1 << shiftLogN
        var shiftEvals = [Fr](repeating: Fr.zero, count: shiftN)
        for i in 0..<shiftN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            shiftEvals[i] = frFromInt(rng >> 32)
        }
        let alpha = frFromInt(7)

        let gpuShifted = try engine.domainShift(evals: shiftEvals, alpha: alpha, logN: shiftLogN)
        let cpuShifted = STIREngine.cpuDomainShift(evals: shiftEvals, alpha: alpha, logN: shiftLogN)

        var shiftCorrect = gpuShifted.count == cpuShifted.count
        if shiftCorrect {
            for i in 0..<gpuShifted.count {
                if frToInt(gpuShifted[i]) != frToInt(cpuShifted[i]) {
                    print("  DOMAIN SHIFT MISMATCH at \(i)")
                    shiftCorrect = false
                    break
                }
            }
        }
        print("  Domain shift 2^\(shiftLogN): \(shiftCorrect ? "PASS" : "FAIL")")

        // --- Correctness: full STIR commit + query + verify ---
        print("\n--- STIR protocol correctness ---")
        let protoLogN = 12
        let protoN = 1 << protoLogN
        var protoEvals = [Fr](repeating: Fr.zero, count: protoN)
        for i in 0..<protoN {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            protoEvals[i] = frFromInt(rng >> 32)
        }

        var betas = [Fr]()
        var shifts = [Fr]()
        for i in 0..<protoLogN {
            betas.append(frFromInt(UInt64(i + 1) * 17))
            shifts.append(frFromInt(UInt64(i + 1) * 31))
        }

        let commitment = try engine.commitPhase(evals: protoEvals, betas: betas, shifts: shifts)
        print("  Commit: \(commitment.layers.count) layers, \(commitment.roots.count) roots")

        let queryIndices: [UInt32] = [0, 42, 1000, UInt32(protoN / 2 - 1)]
        let queries = try engine.queryPhase(commitment: commitment, queryIndices: queryIndices)
        print("  Query: \(queries.count) proofs extracted")

        let verified = engine.verify(commitment: commitment, queries: queries)
        print("  Verify: \(verified ? "PASS" : "FAIL")")

        // --- Soundness comparison: STIR vs FRI ---
        print("\n--- Soundness comparison (queries needed for 128-bit security) ---")
        let rates = [0.25, 0.125, 0.0625]
        for rate in rates {
            let friQ = STIREngine.queriesNeeded(securityBits: 128, rate: rate, useSTIR: false)
            let stirQ = STIREngine.queriesNeeded(securityBits: 128, rate: rate, useSTIR: true)
            let saving = 100.0 * Double(friQ - stirQ) / Double(friQ)
            print(String(format: "  rate=1/%-3d | FRI: %3d queries | STIR: %3d queries | %.0f%% fewer",
                        Int(1.0 / rate), friQ, stirQ, saving))
        }

        // --- Proof size comparison ---
        print("\n--- Proof size comparison ---")
        for logN in [14, 16, 18, 20] {
            let rate = 0.25
            let friQ = STIREngine.queriesNeeded(securityBits: 128, rate: rate, useSTIR: false)
            let stirQ = STIREngine.queriesNeeded(securityBits: 128, rate: rate, useSTIR: true)
            let friSize = STIREngine.estimateProofSize(logN: logN, numQueries: friQ, rate: rate)
            let stirSize = STIREngine.estimateProofSize(logN: logN, numQueries: stirQ, rate: rate)
            let saving = 100.0 * Double(friSize - stirSize) / Double(friSize)
            print(String(format: "  2^%-2d | FRI: %5.1fKB (%d queries) | STIR: %5.1fKB (%d queries) | %.0f%% smaller",
                        logN, Double(friSize) / 1024.0, friQ,
                        Double(stirSize) / 1024.0, stirQ, saving))
        }

        // --- Performance: domain shift ---
        print("\n--- Domain shift performance ---")
        let shiftSizes = [12, 14, 16, 18]
        for logN in shiftSizes {
            let n = 1 << logN
            var evals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = frFromInt(rng >> 32)
            }
            let a = frFromInt(42)

            // Warmup
            let _ = try engine.domainShift(evals: evals, alpha: a, logN: logN)

            var times = [Double]()
            for _ in 0..<10 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.domainShift(evals: evals, alpha: a, logN: logN)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[5]
            print(String(format: "  2^%-2d: %7.2fms (iNTT + shift + NTT)", logN, median))
        }

        // --- Performance: STIR commit phase ---
        print("\n--- STIR commit phase performance ---")
        for logN in [12, 14, 16] {
            let n = 1 << logN
            var evals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = frFromInt(rng >> 32)
            }
            var b = [Fr]()
            var s = [Fr]()
            for i in 0..<logN {
                b.append(frFromInt(UInt64(i + 1) * 7))
                s.append(frFromInt(UInt64(i + 1) * 13))
            }

            // Warmup
            let _ = try engine.commitPhase(evals: evals, betas: b, shifts: s)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try engine.commitPhase(evals: evals, betas: b, shifts: s)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let stirMs = times[2]

            // Compare with FRI commit (fold only, no shift)
            let _ = try friEngine.commitPhase(evals: evals, betas: b)
            var friTimes = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let _ = try friEngine.commitPhase(evals: evals, betas: b)
                friTimes.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            friTimes.sort()
            let friMs = friTimes[2]

            let overhead = stirMs / friMs
            print(String(format: "  2^%-2d | STIR: %7.1fms | FRI: %7.1fms | overhead: %.1fx",
                        logN, stirMs, friMs, overhead))
        }

        // --- Performance: full STIR prove ---
        print("\n--- Full STIR prove (commit + query + verify) ---")
        for logN in [12, 14] {
            let n = 1 << logN
            var evals = [Fr](repeating: Fr.zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1442695040888963407
                evals[i] = frFromInt(rng >> 32)
            }

            // Warmup
            let _ = try engine.prove(evals: evals, numQueries: 4)

            var times = [Double]()
            for _ in 0..<5 {
                let t0 = CFAbsoluteTimeGetCurrent()
                let (comm, proof) = try engine.prove(evals: evals, numQueries: 4)
                let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
                times.append(elapsed)

                // Verify on last iteration
                if times.count == 5 {
                    let ok = engine.verify(commitment: comm, queries: proof.queries)
                    print(String(format: "  2^%-2d | prove: %7.1fms | proof: %dB | verify: %@",
                                logN, times.sorted()[2], proof.sizeBytes, ok ? "PASS" : "FAIL"))
                }
            }
        }

    } catch {
        print("Error: \(error)")
    }
}
