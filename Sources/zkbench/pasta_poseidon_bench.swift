// Pasta Poseidon GPU Benchmark
// Mina Kimchi variant: 55 full rounds, x^7 S-box, full MDS, width=3
import Foundation
import zkMetal

public func runPastaPoseidonBench() {
    print("\n=== Pasta Poseidon GPU Benchmark ===")

    do {
        let engine = try PastaPoseidonEngine()
        let zero = PallasFp.zero

        print("\n--- Pallas Hash Pairs (GPU, baseline) ---")
        let sizes = [1024, 4096, 16384, 65536, 131072, 262144]
        for size in sizes {
            var input = [PallasFp]()
            input.reserveCapacity(size * 2)
            for _ in 0..<(size * 2) { input.append(zero) }

            // Warmup
            _ = try engine.pallasHashPairs(input)

            // Benchmark
            let runs = 5
            var times = [Double]()
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                _ = try engine.pallasHashPairs(input)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[runs / 2]
            let hashesPerSec = Double(size) / (median / 1000)
            print(String(format: "  n=%-7d  %7.2fms  %9.0f hashes/sec", size, median, hashesPerSec))
        }

        // Test batched kernel with different batch sizes at 65536
        print("\n--- Pallas Hash Pairs Batched (GPU, batch size sweep) ---")
        let batchSizes = [1, 2, 4, 8, 16]
        let testSize = 65536
        var testInput = [PallasFp]()
        testInput.reserveCapacity(testSize * 2)
        for _ in 0..<(testSize * 2) { testInput.append(zero) }

        for batchSize in batchSizes {
            // Warmup
            _ = try engine.pallasHashPairsBatched(testInput, batchSize: batchSize)

            // Benchmark
            let runs = 5
            var times = [Double]()
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                _ = try engine.pallasHashPairsBatched(testInput, batchSize: batchSize)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[runs / 2]
            let hashesPerSec = Double(testSize) / (median / 1000)
            print(String(format: "  batchSize=%-2d  %7.2fms  %9.0f hashes/sec", batchSize, median, hashesPerSec))
        }

        print("\n--- Pallas Batch Permute (GPU) ---")
        for size in sizes {
            var input = [PallasFp]()
            input.reserveCapacity(size * 3)
            for _ in 0..<(size * 3) { input.append(zero) }

            // Warmup
            _ = try engine.pallasBatchPermute(input)

            // Benchmark
            let runs = 5
            var times = [Double]()
            for _ in 0..<runs {
                let t0 = CFAbsoluteTimeGetCurrent()
                _ = try engine.pallasBatchPermute(input)
                times.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
            }
            times.sort()
            let median = times[runs / 2]
            let statesPerSec = Double(size) / (median / 1000)
            print(String(format: "  n=%-7d  %7.2fms  %9.0f states/sec", size, median, statesPerSec))
        }

        print("\n--- CPU Baseline (C CIOS) ---")
        for size in sizes {
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0..<size {
                _ = pallasPoseidonHash(zero, zero)
            }
            let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            let hashesPerSec = Double(size) / (elapsed / 1000)
            print(String(format: "  n=%-7d  %7.2fms  %9.0f hashes/sec", size, elapsed, hashesPerSec))
        }

    } catch {
        print("  ERROR: \(error)")
    }
}
