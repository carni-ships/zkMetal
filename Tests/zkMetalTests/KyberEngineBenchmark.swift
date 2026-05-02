// GPU Kyber Engine Benchmark
// Measures the performance improvement from GPU-accelerated NTT integration

import Foundation
import zkMetal

// MARK: - CPU reference (old implementation without GPU NTT)

public func runKyberEngineBenchmark() {
    fputs("\n=== CPU-only Kyber (reference) ===\n", stderr)

    // Create a dummy nttEngine for API compatibility
    guard let nttEngine = (try? LatticeNTTEngine()) else {
        fputs("Failed to init LatticeNTTEngine\n", stderr)
        return
    }

    let kyber = KyberEngine(nttEngine: nttEngine)
    let runs = 10

    // Warmup
    _ = try? kyber.keyGen()

    // KeyGen benchmark
    var times = [Double]()
    for _ in 0..<runs {
        let start = CFAbsoluteTimeGetCurrent()
        _ = try? kyber.keyGen()
        times.append(CFAbsoluteTimeGetCurrent() - start)
    }
    times.sort()
    let median = times[runs / 2] * 1000
    fputs("  KeyGen:       \(String(format: "%8.2f", median)) ms\n", stderr)

    // Encaps benchmark
    if let sk = try? kyber.keyGen() {
        if let (ct, ss) = try? kyber.encapsulate(pk: sk.publicKey) {
            times = []
            for _ in 0..<runs {
                let start = CFAbsoluteTimeGetCurrent()
                _ = try? kyber.encapsulate(pk: sk.publicKey)
                times.append(CFAbsoluteTimeGetCurrent() - start)
            }
            times.sort()
            let encMedian = times[runs / 2] * 1000
            fputs("  Encapsulate:  \(String(format: "%8.2f", encMedian)) ms\n", stderr)

            // Decaps benchmark
            times = []
            for _ in 0..<runs {
                let start = CFAbsoluteTimeGetCurrent()
                _ = try? kyber.decapsulate(sk: sk, ct: ct)
                times.append(CFAbsoluteTimeGetCurrent() - start)
            }
            times.sort()
            let decMedian = times[runs / 2] * 1000
            fputs("  Decapsulate:  \(String(format: "%8.2f", decMedian)) ms\n", stderr)

            // Correctness check
            let recovered = try! kyber.decapsulate(sk: sk, ct: ct)
            fputs("  Correctness:  \(ss == recovered ? "PASS" : "FAIL")\n", stderr)
        }
    }

    // GPU NTT batch performance
    fputs("\n=== GPU Lattice NTT Batch Performance ===\n", stderr)

    fputs("GPU: \(nttEngine.device.name)\n\n", stderr)

    let batchSizes = [10, 100, 1000]

    for batchSize in batchSizes {
        let kFlat = [UInt16](repeating: 42, count: batchSize * 256)

        // Warmup
        _ = try? nttEngine.batchKyberNTT(kFlat, numPolys: batchSize)

        let runs = 5
        var times = [Double]()
        for _ in 0..<runs {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try? nttEngine.batchKyberNTT(kFlat, numPolys: batchSize)
            times.append(CFAbsoluteTimeGetCurrent() - start)
        }
        times.sort()
        let median = times[runs / 2]
        let throughput = Double(batchSize) / median

        fputs("  batch=\(String(format: "%5d", batchSize)): \(String(format: "%8.0f", throughput)) NTTs/s (\(String(format: "%.2f", median * 1000))ms)\n", stderr)
    }

    fputs("\n", stderr)
}