// Blaze SNARK Benchmark — Interleaved RAA Codes
// Validates Blaze engine performance: interleaved encoding, single FRI round, proving

import Foundation
import zkMetal

public func runBlazeBench() {
    fputs("=== Blaze SNARK Benchmark ===\n", stderr)
    fputs("Version: \(BlazeEngine.version.description)\n\n", stderr)

    do {
        let engine = try BlazeEngine(config: .bn254Default)
        let n = engine.config.domainSize
        let m = engine.config.numPolynomials
        let logN = Int(log2(Double(n)))

        fputs(String(format: "Config: n=2^%d (%d), m=%d polynomials, listSize=%d\n",
                    logN, n, m, engine.config.listSize), stderr)
        fputs(String(format: "FRI fold mode: %@\n", engine.config.friFoldMode.rawValue), stderr)
        fputs("GPU: \(engine.device.name)\n\n", stderr)

        // Generate test polynomials
        var rng: UInt64 = 0xDEADBEEFCAFEBABE
        var polys = [[Fr]]()
        for j in 0..<m {
            var poly = [Fr](repeating: .zero, count: n)
            for i in 0..<n {
                rng = rng &* 6364136223846793005 &+ 1
                poly[i] = frFromInt(UInt64(truncatingIfNeeded: rng))
            }
            polys.append(poly)
        }

        // Challenges
        var challenges = [Fr]()
        for i in 0..<10 {
            rng = rng &* 6364136223846793005 &+ 1
            challenges.append(frFromInt(UInt64(truncatingIfNeeded: rng)))
        }

        // ========== Interleaved Encoding Benchmark ==========
        fputs("--- Interleaved Encoding ---\n", stderr)

        var encodeTimes = [Double]()
        for _ in 0..<5 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let codeword = try engine.encodeInterleaved(polys: polys)
            let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            encodeTimes.append(dt)
            _ = codeword  // prevent optimization
        }
        encodeTimes.sort()
        let encodeMedian = encodeTimes[2]
        fputs(String(format: "  Encode (n=%d, m=%d):  %.2f ms (median of 5)\n", n, m, encodeMedian), stderr)

        // ========== Commitment Benchmark ==========
        fputs("\n--- Commitment Phase ---\n", stderr)

        var commitTimes = [Double]()
        var proof: BlazeProof?
        for _ in 0..<3 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let (_, _) = try engine.commit(polys: polys)
            let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            commitTimes.append(dt)
        }
        commitTimes.sort()
        let commitMedian = commitTimes[1]
        fputs(String(format: "  Commit:                %.2f ms (median of 3)\n", commitMedian), stderr)

        // ========== Full Prove Benchmark ==========
        fputs("\n--- Prove Benchmark ---\n", stderr)

        var proveTimes = [Double]()
        for i in 0..<3 {
            let t0 = CFAbsoluteTimeGetCurrent()
            let p = try engine.prove(polys: polys)
            let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            proveTimes.append(dt)
            if i == 0 { proof = p }
            fputs(String(format: "  Run %d: %.2f ms\n", i + 1, dt), stderr)
        }
        proveTimes.sort()
        let proveMedian = proveTimes[1]
        fputs(String(format: "  Median: %.2f ms\n", proveMedian), stderr)

        // ========== Proof Stats ==========
        if let proof = proof {
            fputs("\n--- Proof Statistics ---\n", stderr)
            fputs(String(format: "  Codeword root size:   %d bytes\n", proof.codewordRoot.count), stderr)
            fputs(String(format: "  FRI final evals:     %d\n", proof.friProof.foldedEvals.count), stderr)
            fputs(String(format: "  Query indices:        %d\n", proof.queryIndices.count), stderr)
            fputs(String(format: "  Query openings:       %d x %d\n",
                        proof.queryOpenings.count, proof.queryOpenings[0].count), stderr)
            fputs(String(format: "  Estimated proof size: %d bytes\n", proof.estimatedSizeBytes), stderr)
        }

        // ========== Comparison with Traditional STARK ==========
        fputs("\n--- Comparison ---\n", stderr)
        let starkLayers = logN  // Traditional STARK: logN FRI rounds
        let blazeRounds = 1     // Blaze: single round
        fputs(String(format: "  Traditional STARK:     %d FRI rounds\n", starkLayers), stderr)
        fputs(String(format: "  Blaze:                 %d FRI round\n", blazeRounds), stderr)
        fputs(String(format: "  Round reduction:       %.1fx\n", Double(starkLayers) / Double(blazeRounds)), stderr)

        let starkRoots = m  // Traditional: m Merkle roots
        let blazeRoots = 1  // Blaze: 1 root
        fputs(String(format: "  Traditional STARK:     %d Merkle roots\n", starkRoots), stderr)
        fputs(String(format: "  Blaze:                 %d Merkle root\n", blazeRoots), stderr)

        fputs("\n  Note: Blaze proof structure is different - combines all polynomials\n", stderr)
        fputs("        into single codeword for 1-round proving.\n", stderr)

    } catch {
        fputs("Error: \(error)\n", stderr)
    }
}
