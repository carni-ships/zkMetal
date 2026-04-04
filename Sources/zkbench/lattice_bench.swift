// Lattice cryptography benchmarks — Kyber KEM and Dilithium signatures
// Measures GPU-accelerated post-quantum crypto on Apple Metal

import Foundation
import zkMetal

public func runLatticeBench() {
    fputs("\n=== Lattice Cryptography Benchmark ===\n", stderr)
    fputs("Kyber-768 (q=3329, 16-bit) + Dilithium2 (q=8380417, 32-bit)\n\n", stderr)

    // Run field benchmarks first (CPU-only, no GPU needed)
    runFieldBenchmarks()

    do {
        fputs("Initializing GPU LatticeNTTEngine...\n", stderr)
        let nttEngine = try LatticeNTTEngine()
        fputs("GPU: \(nttEngine.device.name)\n\n", stderr)

        runNTTBenchmarks(nttEngine: nttEngine)
        runKyberBenchmarks(nttEngine: nttEngine)
        runDilithiumBenchmarks(nttEngine: nttEngine)
    } catch {
        fputs("GPU Error: \(error)\n", stderr)
    }
}

// MARK: - Field Arithmetic Benchmarks

private func runFieldBenchmarks() {
    fputs("--- Field Arithmetic (CPU) ---\n", stderr)

    // Kyber field
    let iters = 1_000_000
    var a = KyberField(value: 1234)
    var b = KyberField(value: 2345)

    var t0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iters {
        a = kyberMul(a, b)
        b = kyberAdd(b, a)
    }
    var elapsed = CFAbsoluteTimeGetCurrent() - t0
    fputs("  Kyber mul+add:     \(String(format: "%.1f", Double(iters) / elapsed / 1e6)) Mops/s  (sink: \(a.value))\n", stderr)
    fflush(stderr)

    // Dilithium field
    fputs("  Starting Dilithium field bench...\n", stderr)
    fflush(stderr)
    var c = DilithiumField(value: 123456)
    var d = DilithiumField(value: 7654321)

    t0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<iters {
        c = dilithiumMul(c, d)
        d = dilithiumAdd(d, c)
    }
    elapsed = CFAbsoluteTimeGetCurrent() - t0
    fputs("  Dilithium mul+add: \(String(format: "%.1f", Double(iters) / elapsed / 1e6)) Mops/s  (sink: \(c.value))\n", stderr)
    fputs("\n", stderr)
}

// MARK: - NTT Benchmarks

private func runNTTBenchmarks(nttEngine: LatticeNTTEngine) {
    fputs("--- NTT Benchmarks (256-element polynomials) ---\n", stderr)

    // CPU NTT
    let cpuIters = 10000
    var poly = (0..<256).map { KyberField(value: UInt16($0 % Int(KyberField.Q))) }

    var t0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<cpuIters {
        kyberNTTCPU(&poly)
    }
    var elapsed = CFAbsoluteTimeGetCurrent() - t0
    fputs("  Kyber CPU NTT:       \(String(format: "%.0f", Double(cpuIters) / elapsed)) NTTs/s\n", stderr)

    var dpoly = (0..<256).map { DilithiumField(value: UInt32($0)) }
    t0 = CFAbsoluteTimeGetCurrent()
    for _ in 0..<cpuIters {
        dilithiumNTTCPU(&dpoly)
    }
    elapsed = CFAbsoluteTimeGetCurrent() - t0
    fputs("  Dilithium CPU NTT:   \(String(format: "%.0f", Double(cpuIters) / elapsed)) NTTs/s\n", stderr)

    // GPU batch NTT
    fputs("\n  GPU batch NTT throughput (batch_size -> NTTs/s):\n", stderr)
    let batchSizes = [100, 1000, 10000]

    for batchSize in batchSizes {
        do {
            // Kyber GPU NTT
            let kFlat = [UInt16](repeating: 42, count: batchSize * 256)

            // Warmup
            let _ = try nttEngine.batchKyberNTT(kFlat, numPolys: batchSize)

            let runs = 5
            var times = [Double]()
            for _ in 0..<runs {
                let start = CFAbsoluteTimeGetCurrent()
                let _ = try nttEngine.batchKyberNTT(kFlat, numPolys: batchSize)
                times.append(CFAbsoluteTimeGetCurrent() - start)
            }
            times.sort()
            let kyberMedian = times[runs / 2]
            let kyberThroughput = Double(batchSize) / kyberMedian

            // Dilithium GPU NTT
            let dFlat = [UInt32](repeating: 42, count: batchSize * 256)
            let _ = try nttEngine.batchDilithiumNTT(dFlat, numPolys: batchSize)

            times = []
            for _ in 0..<runs {
                let start = CFAbsoluteTimeGetCurrent()
                let _ = try nttEngine.batchDilithiumNTT(dFlat, numPolys: batchSize)
                times.append(CFAbsoluteTimeGetCurrent() - start)
            }
            times.sort()
            let dilMedian = times[runs / 2]
            let dilThroughput = Double(batchSize) / dilMedian

            fputs("    batch=\(String(format: "%5d", batchSize)):  Kyber \(String(format: "%8.0f", kyberThroughput)) NTTs/s  (\(String(format: "%.2f", kyberMedian * 1000))ms)", stderr)
            fputs("  Dilithium \(String(format: "%8.0f", dilThroughput)) NTTs/s  (\(String(format: "%.2f", dilMedian * 1000))ms)\n", stderr)
        } catch {
            fputs("    batch=\(batchSize): Error: \(error)\n", stderr)
        }
    }
    fputs("\n", stderr)
}

// MARK: - Kyber Benchmarks

private func runKyberBenchmarks(nttEngine: LatticeNTTEngine) {
    fputs("--- Kyber-768 KEM ---\n", stderr)

    let kyber = KyberEngine(nttEngine: nttEngine)
    let runs = 20

    // KeyGen
    do {
        // Warmup
        let _ = try kyber.keyGen()

        var times = [Double]()
        var lastSK: KyberSecretKey?
        for _ in 0..<runs {
            let start = CFAbsoluteTimeGetCurrent()
            let sk = try kyber.keyGen()
            times.append(CFAbsoluteTimeGetCurrent() - start)
            lastSK = sk
        }
        times.sort()
        let median = times[runs / 2] * 1000
        fputs("  KeyGen:       \(String(format: "%8.2f", median)) ms  (\(String(format: "%.0f", 1000.0 / median)) ops/s)\n", stderr)

        // Encapsulate
        guard let sk = lastSK else { return }
        let _ = try kyber.encapsulate(pk: sk.publicKey)

        times = []
        var lastCT: KyberCiphertext?
        var lastSS: [UInt8]?
        for _ in 0..<runs {
            let start = CFAbsoluteTimeGetCurrent()
            let (ct, ss) = try kyber.encapsulate(pk: sk.publicKey)
            times.append(CFAbsoluteTimeGetCurrent() - start)
            lastCT = ct
            lastSS = ss
        }
        times.sort()
        let encMedian = times[runs / 2] * 1000
        fputs("  Encapsulate:  \(String(format: "%8.2f", encMedian)) ms  (\(String(format: "%.0f", 1000.0 / encMedian)) ops/s)\n", stderr)

        // Decapsulate
        guard let ct = lastCT, let ss = lastSS else { return }
        let recovered = try kyber.decapsulate(sk: sk, ct: ct)

        times = []
        for _ in 0..<runs {
            let start = CFAbsoluteTimeGetCurrent()
            let _ = try kyber.decapsulate(sk: sk, ct: ct)
            times.append(CFAbsoluteTimeGetCurrent() - start)
        }
        times.sort()
        let decMedian = times[runs / 2] * 1000
        fputs("  Decapsulate:  \(String(format: "%8.2f", decMedian)) ms  (\(String(format: "%.0f", 1000.0 / decMedian)) ops/s)\n", stderr)

        // Correctness check
        let match = (ss == recovered)
        fputs("  Correctness:  \(match ? "PASS" : "FAIL") (shared secrets \(match ? "match" : "MISMATCH"))\n", stderr)

    } catch {
        fputs("  Error: \(error)\n", stderr)
    }
    fputs("\n", stderr)
}

// MARK: - Dilithium Benchmarks

private func runDilithiumBenchmarks(nttEngine: LatticeNTTEngine) {
    fputs("--- Dilithium2 (ML-DSA-44) Signatures ---\n", stderr)

    let dil = DilithiumEngine(nttEngine: nttEngine)
    let runs = 10

    do {
        // KeyGen
        let _ = try dil.keyGen()

        var times = [Double]()
        var lastSK: DilithiumSecretKey?
        for _ in 0..<runs {
            let start = CFAbsoluteTimeGetCurrent()
            let sk = try dil.keyGen()
            times.append(CFAbsoluteTimeGetCurrent() - start)
            lastSK = sk
        }
        times.sort()
        let kgMedian = times[runs / 2] * 1000
        fputs("  KeyGen:   \(String(format: "%8.2f", kgMedian)) ms  (\(String(format: "%.0f", 1000.0 / kgMedian)) ops/s)\n", stderr)

        guard let sk = lastSK else { return }
        let message: [UInt8] = Array("Hello, post-quantum world!".utf8)

        // Sign
        let _ = try dil.sign(sk: sk, message: message)

        times = []
        var lastSig: DilithiumSignature?
        for _ in 0..<runs {
            let start = CFAbsoluteTimeGetCurrent()
            let sig = try dil.sign(sk: sk, message: message)
            times.append(CFAbsoluteTimeGetCurrent() - start)
            lastSig = sig
        }
        times.sort()
        let signMedian = times[runs / 2] * 1000
        fputs("  Sign:     \(String(format: "%8.2f", signMedian)) ms  (\(String(format: "%.0f", 1000.0 / signMedian)) ops/s)\n", stderr)

        // Verify
        guard let sig = lastSig else { return }
        let valid = try dil.verify(pk: sk.publicKey, message: message, signature: sig)

        times = []
        for _ in 0..<runs {
            let start = CFAbsoluteTimeGetCurrent()
            let _ = try dil.verify(pk: sk.publicKey, message: message, signature: sig)
            times.append(CFAbsoluteTimeGetCurrent() - start)
        }
        times.sort()
        let verifyMedian = times[runs / 2] * 1000
        fputs("  Verify:   \(String(format: "%8.2f", verifyMedian)) ms  (\(String(format: "%.0f", 1000.0 / verifyMedian)) ops/s)\n", stderr)

        fputs("  Correctness:  \(valid ? "PASS" : "FAIL") (signature \(valid ? "valid" : "INVALID"))\n", stderr)

    } catch {
        fputs("  Error: \(error)\n", stderr)
    }
    fputs("\n", stderr)
}
