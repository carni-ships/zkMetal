// Binary FRI Benchmark
//
// Benchmarks for binary-native FRI implementation.

import Foundation
import zkMetal

// MARK: - Binary FRI Benchmark Results

struct BinaryFRIBenchResult {
    let logDomainSize: Int
    let numRounds: Int
    let foldTimeMs: Double
    let merkleTimeMs: Double
    let totalTimeMs: Double
}

func formatBinaryFRIBenchTable(results: [BinaryFRIBenchResult]) -> String {
    var table = "Binary FRI Benchmark Results\n"
    table += "===========================\n\n"
    table += String(format: "%-12s %-10s %-12s %-14s %-12s\n",
                   "Log Domain", "Rounds", "Fold (ms)", "Merkle (ms)", "Total (ms)")
    table += String(repeating: "-", count: 60) + "\n"

    for r in results {
        table += String(format: "%-12d %-10d %-12.2f %-14.2f %-12.2f\n",
                       r.logDomainSize, r.numRounds, r.foldTimeMs, r.merkleTimeMs, r.totalTimeMs)
    }

    return table
}

// MARK: - Binary FRI Benchmark

func runBinaryFRIBench() {
    var results = [BinaryFRIBenchResult]()

    // Test configurations
    let configs: [(logSize: Int, finalDegree: Int)] = [
        (8, 3),
        (10, 3),
        (12, 3),
        (14, 3),
    ]

    print("\n=== Binary FRI Benchmark ===\n")

    for (logSize, finalDegree) in configs {
        let config = BinaryFRIConfig(
            foldingFactor: 2,
            numQueries: 16,
            finalPolyMaxDegree: finalDegree,
            logDomainSize: logSize
        )

        let domainSize = 1 << logSize
        print("Testing logDomainSize=\(logSize) (\(domainSize) points)...")

        // Generate evaluations
        let evals = (0..<domainSize).map { UInt8($0) }

        // Create prover
        let prover = BinaryFRIProver(config: config)

        // Compute number of rounds
        let numRounds = prover.computeNumRounds(logSize: logSize)

        // Generate alphas
        var alphas = [UInt8]()
        for i in 0..<numRounds {
            alphas.append(UInt8(i + 1))
        }

        // Benchmark
        let t0 = CFAbsoluteTimeGetCurrent()

        do {
            let (_, witness) = try prover.prove(evals: evals, alphas: alphas)
            let layers = witness.layerEvals

            let foldTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000

            // Build Merkle trees for each layer
            var merkleTime = CFAbsoluteTimeGetCurrent()
            var merkleRoots = [BinaryFRIMerkleCommitment]()

            for layer in layers {
                let logLeaves = layer.count > 1 ? Int(log2(Double(layer.count))) : 0
                let merkleParams = BinaryMerkleParams(logLeaves: logLeaves)
                let tree = BinaryMerkleTree(evaluations: layer, params: merkleParams)
                merkleRoots.append(BinaryFRIMerkleCommitment(
                    root: tree.root,
                    numLeaves: layer.count
                ))
            }

            merkleTime = (CFAbsoluteTimeGetCurrent() - merkleTime) * 1000

            let totalTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000

            let result = BinaryFRIBenchResult(
                logDomainSize: logSize,
                numRounds: numRounds,
                foldTimeMs: foldTime,
                merkleTimeMs: merkleTime,
                totalTimeMs: totalTime
            )
            results.append(result)

            print("  -> \(numRounds) rounds, fold: \(String(format: "%.2f", foldTime))ms, merkle: \(String(format: "%.2f", merkleTime))ms, total: \(String(format: "%.2f", totalTime))ms")
        } catch {
            print("  ERROR: \(error)")
        }
    }

    // Print summary table
    if !results.isEmpty {
        print("\n" + formatBinaryFRIBenchTable(results: results))
    }
}

// MARK: - GPU Binary FRI Benchmark

func runGPUBinaryFRIBench() {
    print("\n=== GPU Binary FRI Benchmark ===\n")

    // Try to create GPU engine
    let engine: GPUBinaryFRIFoldEngine?
    do {
        engine = try GPUBinaryFRIFoldEngine()
        print("GPU engine created successfully")
    } catch {
        print("GPU not available: \(error)")
        return
    }

    guard let gpuEngine = engine else { return }

    // Test configurations
    let configs: [(logSize: Int, finalDegree: Int)] = [
        (8, 3),
        (10, 3),
        (12, 3),
    ]

    for (logSize, finalDegree) in configs {
        let config = BinaryFRIConfig(
            foldingFactor: 2,
            numQueries: 16,
            finalPolyMaxDegree: finalDegree,
            logDomainSize: logSize
        )

        let domainSize = 1 << logSize
        print("Testing logDomainSize=\(logSize) (\(domainSize) points)...")

        // Generate random evaluations
        let evals = (0..<domainSize).map { _ in UInt8.random(in: 0...255) }

        // Benchmark GPU folding
        let t0 = CFAbsoluteTimeGetCurrent()

        let prover = BinaryFRIProver(config: config, gpuEngine: gpuEngine)
        let numRounds = prover.computeNumRounds(logSize: logSize)

        var alphas = [UInt8]()
        for i in 0..<numRounds {
            alphas.append(UInt8(i + 1))
        }

        do {
            let (key, _) = try prover.prove(evals: evals, alphas: alphas)
            let totalTime = (CFAbsoluteTimeGetCurrent() - t0) * 1000

            print("  -> \(key.numRounds) rounds, total: \(String(format: "%.2f", totalTime))ms")
        } catch {
            print("  -> Error: \(error)")
        }
    }
}
