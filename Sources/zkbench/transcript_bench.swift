// Transcript Benchmark — Fiat-Shamir transcript throughput and correctness

import Foundation
import zkMetal

/// Compare two Fr elements for equality (compare Montgomery-form limbs directly)
private func frEqual(_ a: Fr, _ b: Fr) -> Bool {
    return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3
        && a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

public func runTranscriptBench() {
    fputs("\n--- Transcript Benchmark ---\n", stderr)

    // Generate test Fr elements
    let count = 1000
    var elems = [Fr]()
    elems.reserveCapacity(count)
    var rng: UInt64 = 0xCAFE_BABE_DEAD_BEEF
    for _ in 0..<count {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        elems.append(frFromInt(rng))
    }

    // --- Throughput: absorb + squeeze cycles ---
    for backend in [Transcript.HashBackend.poseidon2, .keccak256] {
        let name = backend == .poseidon2 ? "Poseidon2" : "Keccak-256"

        // Warmup
        let warmup = Transcript(label: "warmup", backend: backend)
        for e in elems.prefix(100) { warmup.absorb(e) }
        _ = warmup.squeeze()

        // Benchmark: absorb N elements, squeeze N challenges
        let runs = 5
        var times = [Double]()
        for _ in 0..<runs {
            let t = Transcript(label: "bench", backend: backend)
            let start = CFAbsoluteTimeGetCurrent()
            for e in elems { t.absorb(e) }
            for _ in 0..<count { _ = t.squeeze() }
            let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
            times.append(elapsed)
        }
        times.sort()
        let median = times[runs / 2]
        let opsPerSec = Double(count * 2) / (median / 1000)  // absorb + squeeze = 2 ops each
        let padded = name.padding(toLength: 10, withPad: " ", startingAt: 0)
        fputs("  \(padded) \(count) absorb + \(count) squeeze: \(String(format: "%.2f", median)) ms  (\(String(format: "%.0f", opsPerSec)) ops/s)\n", stderr)
    }

    // --- Correctness: Determinism ---
    fputs("\n  Correctness tests:\n", stderr)

    for backend in [Transcript.HashBackend.poseidon2, .keccak256] {
        let name = backend == .poseidon2 ? "Poseidon2" : "Keccak-256"

        // Same inputs must produce same outputs
        let t1 = Transcript(label: "determinism-test", backend: backend)
        let t2 = Transcript(label: "determinism-test", backend: backend)
        t1.absorb(elems[0])
        t1.absorb(elems[1])
        t2.absorb(elems[0])
        t2.absorb(elems[1])
        let c1 = t1.squeeze()
        let c2 = t2.squeeze()
        let det = frEqual(c1, c2)
        fputs("    [\(det ? "PASS" : "FAIL")] \(name) determinism\n", stderr)
    }

    // Different labels must produce different outputs
    for backend in [Transcript.HashBackend.poseidon2, .keccak256] {
        let name = backend == .poseidon2 ? "Poseidon2" : "Keccak-256"

        let t1 = Transcript(label: "protocol-A", backend: backend)
        let t2 = Transcript(label: "protocol-B", backend: backend)
        t1.absorb(elems[0])
        t2.absorb(elems[0])
        let c1 = t1.squeeze()
        let c2 = t2.squeeze()
        let sep = !frEqual(c1, c2)
        fputs("    [\(sep ? "PASS" : "FAIL")] \(name) domain separation (different labels)\n", stderr)
    }

    // Different inputs must produce different outputs
    for backend in [Transcript.HashBackend.poseidon2, .keccak256] {
        let name = backend == .poseidon2 ? "Poseidon2" : "Keccak-256"

        let t1 = Transcript(label: "same-label", backend: backend)
        let t2 = Transcript(label: "same-label", backend: backend)
        t1.absorb(elems[0])
        t2.absorb(elems[1])
        let c1 = t1.squeeze()
        let c2 = t2.squeeze()
        let diff = !frEqual(c1, c2)
        fputs("    [\(diff ? "PASS" : "FAIL")] \(name) different inputs -> different challenges\n", stderr)
    }

    // Fork produces different challenges than parent
    for backend in [Transcript.HashBackend.poseidon2, .keccak256] {
        let name = backend == .poseidon2 ? "Poseidon2" : "Keccak-256"

        let parent = Transcript(label: "parent", backend: backend)
        parent.absorb(elems[0])
        let child1 = parent.fork(label: "child-1")
        let child2 = parent.fork(label: "child-2")
        let c1 = child1.squeeze()
        let c2 = child2.squeeze()
        let forkSep = !frEqual(c1, c2)
        fputs("    [\(forkSep ? "PASS" : "FAIL")] \(name) fork separation\n", stderr)
    }

    // Multiple squeezes produce distinct challenges
    for backend in [Transcript.HashBackend.poseidon2, .keccak256] {
        let name = backend == .poseidon2 ? "Poseidon2" : "Keccak-256"

        let t = Transcript(label: "multi-squeeze", backend: backend)
        t.absorb(elems[0])
        let challenges = t.squeezeN(5)
        var allDifferent = true
        for i in 0..<challenges.count {
            for j in (i+1)..<challenges.count {
                if frEqual(challenges[i], challenges[j]) {
                    allDifferent = false
                }
            }
        }
        fputs("    [\(allDifferent ? "PASS" : "FAIL")] \(name) sequential squeezes are distinct\n", stderr)
    }

    // absorbLabel mid-protocol changes challenge stream
    for backend in [Transcript.HashBackend.poseidon2, .keccak256] {
        let name = backend == .poseidon2 ? "Poseidon2" : "Keccak-256"

        let t1 = Transcript(label: "step-test", backend: backend)
        let t2 = Transcript(label: "step-test", backend: backend)
        t1.absorb(elems[0])
        t2.absorb(elems[0])
        t1.absorbLabel("step-A")
        t2.absorbLabel("step-B")
        let c1 = t1.squeeze()
        let c2 = t2.squeeze()
        let stepSep = !frEqual(c1, c2)
        fputs("    [\(stepSep ? "PASS" : "FAIL")] \(name) absorbLabel domain separation\n", stderr)
    }

    // Byte-level absorb roundtrip
    for backend in [Transcript.HashBackend.poseidon2, .keccak256] {
        let name = backend == .poseidon2 ? "Poseidon2" : "Keccak-256"

        let t1 = Transcript(label: "bytes", backend: backend)
        let t2 = Transcript(label: "bytes", backend: backend)
        let data: [UInt8] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        t1.absorbBytes(data)
        t2.absorbBytes(data)
        let c1 = t1.squeeze()
        let c2 = t2.squeeze()
        let bytesDet = frEqual(c1, c2)
        fputs("    [\(bytesDet ? "PASS" : "FAIL")] \(name) absorbBytes determinism\n", stderr)
    }
}
