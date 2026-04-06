// Batch Transcript Benchmark — GPU batch Fiat-Shamir transcript correctness and throughput
//
// Tests:
//   1. Correctness: GPU batch results match individual CPU Poseidon2Sponge transcripts
//   2. Throughput: GPU batch vs sequential CPU for varying batch sizes
//   3. Multi-step: absorb -> squeeze -> absorb -> squeeze pipeline

import Foundation
import zkMetal

/// Compare two Fr elements for equality (Montgomery-form limbs).
private func frEq(_ a: Fr, _ b: Fr) -> Bool {
    return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3
        && a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

/// Generate deterministic test Fr elements from a seed.
private func generateTestElements(count: Int, seed: UInt64 = 0xCAFE_BABE_DEAD_BEEF) -> [Fr] {
    var elems = [Fr]()
    elems.reserveCapacity(count)
    var rng = seed
    for _ in 0..<count {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        elems.append(frFromInt(rng))
    }
    return elems
}

/// CPU reference: absorb message into fresh sponge, squeeze one element.
private func cpuSpongeAbsorbSqueeze(message: [Fr], domainTag: UInt64 = 0, squeezeCount: Int = 1) -> [Fr] {
    var s0 = Fr.zero
    var s1 = Fr.zero
    var s2 = frFromInt(domainTag)
    var absorbed = 0

    for elem in message {
        if absorbed == 0 {
            s0 = frAdd(s0, elem)
        } else {
            s1 = frAdd(s1, elem)
        }
        absorbed += 1
        if absorbed == 2 {
            poseidon2PermuteInPlace(&s0, &s1, &s2)
            absorbed = 0
        }
    }

    // Squeeze
    if absorbed > 0 || !message.isEmpty {
        poseidon2PermuteInPlace(&s0, &s1, &s2)
    }

    var results = [Fr]()
    results.reserveCapacity(squeezeCount)
    var squeezePos = 0
    for _ in 0..<squeezeCount {
        if squeezePos >= 2 {
            poseidon2PermuteInPlace(&s0, &s1, &s2)
            squeezePos = 0
        }
        if squeezePos == 0 {
            results.append(s0)
        } else {
            results.append(s1)
        }
        squeezePos += 1
    }
    return results
}

public func runBatchTranscriptBench() {
    fputs("\n--- GPU Batch Transcript Benchmark ---\n", stderr)

    let engine: GPUBatchTranscript
    do {
        engine = try GPUBatchTranscript()
    } catch {
        fputs("  [SKIP] GPU not available: \(error)\n", stderr)
        return
    }

    let testElems = generateTestElements(count: 4096)

    // ================================================================
    // Correctness Tests
    // ================================================================
    fputs("\n  Correctness tests:\n", stderr)

    // Test 1: Single absorb + single squeeze, uniform messages
    do {
        let n = 128
        let msgLen = 8
        var messages = [[Fr]]()
        for i in 0..<n {
            messages.append(Array(testElems[i * msgLen ..< (i + 1) * msgLen]))
        }

        let gpuResults = try engine.batchAbsorb(values: messages, domainTag: 0)

        var allMatch = true
        for i in 0..<n {
            let cpuResult = cpuSpongeAbsorbSqueeze(message: messages[i], domainTag: 0)
            if !frEq(gpuResults[i], cpuResult[0]) {
                fputs("    Mismatch at index \(i)\n", stderr)
                allMatch = false
                break
            }
        }
        fputs("    [\(allMatch ? "PASS" : "FAIL")] Uniform absorb+squeeze (n=\(n), msgLen=\(msgLen))\n", stderr)
    } catch {
        fputs("    [FAIL] Uniform absorb+squeeze: \(error)\n", stderr)
    }

    // Test 2: Multi-squeeze (squeeze 3 elements per transcript)
    do {
        let n = 128
        let msgLen = 4
        var messages = [[Fr]]()
        for i in 0..<n {
            messages.append(Array(testElems[i * msgLen ..< (i + 1) * msgLen]))
        }
        let squeezeCount = 3

        let states = engine.initStates(count: n, domainTag: 42)
        let afterAbsorb = try engine.absorbUniform(states: states, messages: messages)
        let (_, gpuSqueezed) = try engine.batchSqueeze(states: afterAbsorb, count: squeezeCount)

        var allMatch = true
        for i in 0..<n {
            let cpuResult = cpuSpongeAbsorbSqueeze(message: messages[i], domainTag: 42, squeezeCount: squeezeCount)
            for j in 0..<squeezeCount {
                if !frEq(gpuSqueezed[i][j], cpuResult[j]) {
                    fputs("    Mismatch at transcript \(i), squeeze \(j)\n", stderr)
                    allMatch = false
                    break
                }
            }
            if !allMatch { break }
        }
        fputs("    [\(allMatch ? "PASS" : "FAIL")] Multi-squeeze (n=\(n), squeeze=\(squeezeCount), domainTag=42)\n", stderr)
    } catch {
        fputs("    [FAIL] Multi-squeeze: \(error)\n", stderr)
    }

    // Test 3: Variable-length messages
    do {
        let n = 128
        var messages = [[Fr]]()
        for i in 0..<n {
            let len = (i % 7) + 1  // lengths 1..7
            let start = (i * 7) % (testElems.count - 7)
            messages.append(Array(testElems[start ..< start + len]))
        }

        let states = engine.initStates(count: n, domainTag: 5)
        let afterAbsorb = try engine.absorbVarlen(states: states, messages: messages)
        let (_, gpuSqueezed) = try engine.batchSqueeze(states: afterAbsorb, count: 1)

        var allMatch = true
        for i in 0..<n {
            let cpuResult = cpuSpongeAbsorbSqueeze(message: messages[i], domainTag: 5, squeezeCount: 1)
            if !frEq(gpuSqueezed[i][0], cpuResult[0]) {
                fputs("    Mismatch at index \(i) (msgLen=\(messages[i].count))\n", stderr)
                allMatch = false
                break
            }
        }
        fputs("    [\(allMatch ? "PASS" : "FAIL")] Variable-length absorb (n=\(n), lens=1..7)\n", stderr)
    } catch {
        fputs("    [FAIL] Variable-length absorb: \(error)\n", stderr)
    }

    // Test 4: Multi-step absorb-squeeze-absorb-squeeze
    do {
        let n = 128
        let msgLen = 4

        let messages1 = (0..<n).map { i in Array(testElems[i * msgLen ..< (i + 1) * msgLen]) }
        let messages2 = (0..<n).map { i in Array(testElems[(n + i) * msgLen ..< (n + i + 1) * msgLen]) }

        // GPU path
        var states = engine.initStates(count: n, domainTag: 7)
        states = try engine.absorbUniform(states: states, messages: messages1)
        let (states2, gpuSq1) = try engine.batchSqueeze(states: states, count: 1)
        let states3 = try engine.absorbUniform(states: states2, messages: messages2)
        let (_, gpuSq2) = try engine.batchSqueeze(states: states3, count: 2)

        // CPU reference (manual two-step sponge)
        var allMatch = true
        for i in 0..<n {
            var s0 = Fr.zero, s1 = Fr.zero, s2 = frFromInt(7)
            var abs = 0

            // Step 1: absorb messages1[i]
            for elem in messages1[i] {
                if abs == 0 { s0 = frAdd(s0, elem) } else { s1 = frAdd(s1, elem) }
                abs += 1
                if abs == 2 { poseidon2PermuteInPlace(&s0, &s1, &s2); abs = 0 }
            }

            // Squeeze 1
            if abs > 0 { poseidon2PermuteInPlace(&s0, &s1, &s2); abs = 0 }
            let sq1 = s0
            // After squeeze, mark dirty (need permute before next squeeze)
            // But since we're absorbing next, the state continues from post-permute

            // Step 2: absorb messages2[i] into post-squeeze state
            // The GPU squeeze kernel sets absorbed=0 after squeeze
            for elem in messages2[i] {
                if abs == 0 { s0 = frAdd(s0, elem) } else { s1 = frAdd(s1, elem) }
                abs += 1
                if abs == 2 { poseidon2PermuteInPlace(&s0, &s1, &s2); abs = 0 }
            }

            // Squeeze 2 (two elements)
            if abs > 0 { poseidon2PermuteInPlace(&s0, &s1, &s2); abs = 0 }
            let sq2_0 = s0
            let sq2_1 = s1

            if !frEq(gpuSq1[i][0], sq1) ||
               !frEq(gpuSq2[i][0], sq2_0) ||
               !frEq(gpuSq2[i][1], sq2_1) {
                fputs("    Mismatch at index \(i) in multi-step\n", stderr)
                allMatch = false
                break
            }
        }
        fputs("    [\(allMatch ? "PASS" : "FAIL")] Multi-step absorb-squeeze-absorb-squeeze (n=\(n))\n", stderr)
    } catch {
        fputs("    [FAIL] Multi-step: \(error)\n", stderr)
    }

    // Test 5: Domain separation (different tags produce different outputs)
    do {
        let n = 128
        let msgLen = 4
        let messages = (0..<n).map { i in Array(testElems[i * msgLen ..< (i + 1) * msgLen]) }

        let results1 = try engine.batchAbsorb(values: messages, domainTag: 0)
        let results2 = try engine.batchAbsorb(values: messages, domainTag: 1)

        var allDifferent = true
        for i in 0..<n {
            if frEq(results1[i], results2[i]) {
                allDifferent = false
                break
            }
        }
        fputs("    [\(allDifferent ? "PASS" : "FAIL")] Domain separation (different tags -> different outputs)\n", stderr)
    } catch {
        fputs("    [FAIL] Domain separation: \(error)\n", stderr)
    }

    // Test 6: CPU fallback path (small batch < threshold, should still match)
    do {
        let n = 16  // below gpuThreshold
        let msgLen = 6
        var messages = [[Fr]]()
        for i in 0..<n {
            messages.append(Array(testElems[i * msgLen ..< (i + 1) * msgLen]))
        }

        let gpuResults = try engine.batchAbsorb(values: messages, domainTag: 0)

        var allMatch = true
        for i in 0..<n {
            let cpuResult = cpuSpongeAbsorbSqueeze(message: messages[i], domainTag: 0)
            if !frEq(gpuResults[i], cpuResult[0]) {
                allMatch = false
                break
            }
        }
        fputs("    [\(allMatch ? "PASS" : "FAIL")] CPU fallback path (n=\(n) < threshold)\n", stderr)
    } catch {
        fputs("    [FAIL] CPU fallback: \(error)\n", stderr)
    }

    // ================================================================
    // Throughput Benchmarks
    // ================================================================
    fputs("\n  Throughput benchmarks:\n", stderr)

    let benchElems = generateTestElements(count: 65536)
    let msgLen = 8

    for batchSize in [64, 256, 1024, 4096] {
        let messages = (0..<batchSize).map { i in
            Array(benchElems[(i * msgLen) % (benchElems.count - msgLen) ..< ((i * msgLen) % (benchElems.count - msgLen)) + msgLen])
        }

        // GPU batch
        var gpuTimes = [Double]()
        for _ in 0..<5 {
            let start = CFAbsoluteTimeGetCurrent()
            _ = try? engine.batchAbsorb(values: messages, domainTag: 0)
            gpuTimes.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
        }
        gpuTimes.sort()
        let gpuMedian = gpuTimes[2]

        // Sequential CPU
        var cpuTimes = [Double]()
        for _ in 0..<5 {
            let start = CFAbsoluteTimeGetCurrent()
            for msg in messages {
                _ = cpuSpongeAbsorbSqueeze(message: msg, domainTag: 0)
            }
            cpuTimes.append((CFAbsoluteTimeGetCurrent() - start) * 1000)
        }
        cpuTimes.sort()
        let cpuMedian = cpuTimes[2]

        let speedup = cpuMedian / max(gpuMedian, 0.001)
        fputs("    n=\(String(batchSize).padding(toLength: 5, withPad: " ", startingAt: 0))  GPU: \(String(format: "%7.2f", gpuMedian)) ms  CPU: \(String(format: "%7.2f", cpuMedian)) ms  speedup: \(String(format: "%.1f", speedup))x\n", stderr)
    }

    fputs("  Version: \(GPUBatchTranscript.version.description)\n", stderr)
}
