// GPUPrefixSumEngine tests — GPU vs sequential CPU scan for BN254 Fr

import Foundation
import zkMetal

func runGPUPrefixSumTests() {
    suite("GPUPrefixSum")

    guard let engine = try? GPUPrefixSumEngine() else {
        print("  [SKIP] No GPU available")
        return
    }

    // --- Inclusive scan Fr (small, CPU path) ---
    do {
        let a = Fr.one
        let b = frAdd(Fr.one, Fr.one)  // 2
        let c = frAdd(b, Fr.one)       // 3
        let input = [a, b, c]
        let result = engine.inclusiveScan(values: input)
        expectEqual(result.count, 3, "inclusive Fr count")
        // expected: [1, 1+2=3, 1+2+3=6]
        let e0 = a
        let e1 = frAdd(a, b)
        let e2 = frAdd(frAdd(a, b), c)
        expect(frEqual(result[0], e0), "inclusive Fr[0]")
        expect(frEqual(result[1], e1), "inclusive Fr[1]")
        expect(frEqual(result[2], e2), "inclusive Fr[2]")
        print("  [OK] inclusive scan Fr (small)")
    }

    // --- Exclusive scan Fr (small, CPU path) ---
    do {
        let a = Fr.one
        let b = frAdd(Fr.one, Fr.one)
        let c = frAdd(b, Fr.one)
        let input = [a, b, c]
        let result = engine.exclusiveScan(values: input)
        expectEqual(result.count, 3, "exclusive Fr count")
        // expected: [0, 1, 1+2=3]
        expect(frEqual(result[0], Fr.zero), "exclusive Fr[0] = 0")
        expect(frEqual(result[1], a), "exclusive Fr[1] = a")
        expect(frEqual(result[2], frAdd(a, b)), "exclusive Fr[2] = a+b")
        print("  [OK] exclusive scan Fr (small)")
    }

    // --- Empty input ---
    do {
        expectEqual(engine.inclusiveScan(values: []).count, 0, "empty inclusive Fr")
        expectEqual(engine.exclusiveScan(values: []).count, 0, "empty exclusive Fr")
        expectEqual(engine.segmentedScan(values: [], flags: []).count, 0, "empty segmented Fr")
        print("  [OK] empty inputs")
    }

    // --- Single element ---
    do {
        let input = [Fr.one]
        let incl = engine.inclusiveScan(values: input)
        let excl = engine.exclusiveScan(values: input)
        expect(frEqual(incl[0], Fr.one), "single inclusive = value")
        expect(frEqual(excl[0], Fr.zero), "single exclusive = 0")
        print("  [OK] single element")
    }

    // --- GPU path: inclusive scan with forced GPU ---
    do {
        let n = 1024
        // Create array of Fr.one (so scan should be [1, 2, 3, ..., n])
        let input = [Fr](repeating: Fr.one, count: n)

        engine.cpuThreshold = 0
        let result = engine.inclusiveScan(values: input)
        engine.cpuThreshold = 2048

        expectEqual(result.count, n, "GPU inclusive Fr count")

        // Verify first few
        expect(frEqual(result[0], Fr.one), "GPU inclusive Fr[0] = 1")
        expect(frEqual(result[1], frAdd(Fr.one, Fr.one)), "GPU inclusive Fr[1] = 2")

        // Verify last: sum of n ones = n in Fr
        var expected = Fr.zero
        for _ in 0..<n {
            expected = frAdd(expected, Fr.one)
        }
        expect(frEqual(result[n - 1], expected), "GPU inclusive Fr[last] = n")
        print("  [OK] inclusive scan Fr (GPU, n=\(n))")
    }

    // --- GPU path: exclusive scan ---
    do {
        let n = 1024
        let input = [Fr](repeating: Fr.one, count: n)

        engine.cpuThreshold = 0
        let result = engine.exclusiveScan(values: input)
        engine.cpuThreshold = 2048

        expectEqual(result.count, n, "GPU exclusive Fr count")
        expect(frEqual(result[0], Fr.zero), "GPU exclusive Fr[0] = 0")
        expect(frEqual(result[1], Fr.one), "GPU exclusive Fr[1] = 1")

        // exclusive[n-1] = sum of (n-1) ones
        var expected = Fr.zero
        for _ in 0..<(n - 1) {
            expected = frAdd(expected, Fr.one)
        }
        expect(frEqual(result[n - 1], expected), "GPU exclusive Fr[last]")
        print("  [OK] exclusive scan Fr (GPU, n=\(n))")
    }

    // --- Multi-block scan (larger than one threadgroup) ---
    do {
        let n = 2048  // > 512 elements = more than one block with tg=256
        let input = [Fr](repeating: Fr.one, count: n)

        engine.cpuThreshold = 0
        let result = engine.inclusiveScan(values: input)
        engine.cpuThreshold = 2048

        expectEqual(result.count, n, "multi-block Fr scan count")

        // Verify a few checkpoints
        var ok = true
        var expected = Fr.zero
        for i in 0..<n {
            expected = frAdd(expected, Fr.one)
            if !frEqual(result[i], expected) { ok = false; break }
        }
        expect(ok, "multi-block inclusive scan all-ones")
        print("  [OK] multi-block inclusive scan Fr (n=\(n))")
    }

    // --- GPU vs CPU consistency: inclusive scan ---
    do {
        let n = 1024
        // Use small random-ish values (repeated additions of one/two/three)
        var input = [Fr](repeating: Fr.zero, count: n)
        var val = Fr.one
        for i in 0..<n {
            input[i] = val
            val = frAdd(val, Fr.one)
            if i % 7 == 0 { val = Fr.one }  // reset periodically
        }

        // CPU path
        engine.cpuThreshold = n + 1
        let cpuResult = engine.inclusiveScan(values: input)

        // GPU path
        engine.cpuThreshold = 0
        let gpuResult = engine.inclusiveScan(values: input)
        engine.cpuThreshold = 2048

        var match = true
        for i in 0..<n {
            if !frEqual(cpuResult[i], gpuResult[i]) { match = false; break }
        }
        expect(match, "GPU vs CPU inclusive scan Fr consistency")
        print("  [OK] GPU vs CPU consistency (n=\(n))")
    }

    // --- GPU vs CPU consistency: exclusive scan ---
    do {
        let n = 1024
        var input = [Fr](repeating: Fr.zero, count: n)
        var val = Fr.one
        for i in 0..<n {
            input[i] = val
            val = frAdd(val, Fr.one)
            if i % 5 == 0 { val = Fr.one }
        }

        engine.cpuThreshold = n + 1
        let cpuResult = engine.exclusiveScan(values: input)

        engine.cpuThreshold = 0
        let gpuResult = engine.exclusiveScan(values: input)
        engine.cpuThreshold = 2048

        var match = true
        for i in 0..<n {
            if !frEqual(cpuResult[i], gpuResult[i]) { match = false; break }
        }
        expect(match, "GPU vs CPU exclusive scan Fr consistency")
        print("  [OK] GPU vs CPU exclusive consistency (n=\(n))")
    }

    // --- Segmented scan (CPU path) ---
    do {
        let a = Fr.one
        let b = frAdd(Fr.one, Fr.one)
        let c = frAdd(b, Fr.one)
        // Two segments: [a, b | c, a, b]
        let input = [a, b, c, a, b]
        let flags: [Bool] = [true, false, true, false, false]
        let result = engine.segmentedScan(values: input, flags: flags)

        expectEqual(result.count, 5, "segmented count")
        // Segment 1: [a, a+b]
        expect(frEqual(result[0], a), "seg[0] = a")
        expect(frEqual(result[1], frAdd(a, b)), "seg[1] = a+b")
        // Segment 2: [c, c+a, c+a+b]
        expect(frEqual(result[2], c), "seg[2] = c (new segment)")
        expect(frEqual(result[3], frAdd(c, a)), "seg[3] = c+a")
        expect(frEqual(result[4], frAdd(frAdd(c, a), b)), "seg[4] = c+a+b")
        print("  [OK] segmented scan Fr (small)")
    }

    // --- Segmented scan (GPU path) ---
    do {
        let n = 1024
        let segLen = 64  // segment length
        var input = [Fr](repeating: Fr.one, count: n)
        var flags = [Bool](repeating: false, count: n)

        // Mark segment boundaries
        for i in stride(from: 0, to: n, by: segLen) {
            flags[i] = true
        }

        engine.cpuThreshold = 0
        let gpuResult = engine.segmentedScan(values: input, flags: flags)
        engine.cpuThreshold = 2048

        // CPU reference
        let cpuResult = cpuSegmentedScanRef(input, flags: flags)

        var match = true
        for i in 0..<n {
            if !frEqual(gpuResult[i], cpuResult[i]) { match = false; break }
        }
        expect(match, "GPU vs CPU segmented scan consistency")
        print("  [OK] segmented scan Fr (GPU, n=\(n))")
    }

    // --- Performance: large inclusive scan ---
    do {
        let n = 1 << 16  // 64K elements
        let input = [Fr](repeating: Fr.one, count: n)

        engine.cpuThreshold = 0
        let t0 = CFAbsoluteTimeGetCurrent()
        let result = engine.inclusiveScan(values: input)
        let gpuTime = CFAbsoluteTimeGetCurrent() - t0
        engine.cpuThreshold = 2048

        // Verify last element
        var expected = Fr.zero
        for _ in 0..<n {
            expected = frAdd(expected, Fr.one)
        }
        expect(frEqual(result[n - 1], expected), "64K scan last element")

        // CPU timing
        let t1 = CFAbsoluteTimeGetCurrent()
        engine.cpuThreshold = n + 1
        let _ = engine.inclusiveScan(values: input)
        let cpuTime = CFAbsoluteTimeGetCurrent() - t1
        engine.cpuThreshold = 2048

        print(String(format: "  [OK] 64K Fr inclusive scan: GPU=%.2fms CPU=%.2fms (%.1fx)",
                      gpuTime * 1000, cpuTime * 1000, cpuTime / max(gpuTime, 0.000001)))
    }
}

// MARK: - Helpers

private func frEqual(_ a: Fr, _ b: Fr) -> Bool {
    return a.v.0 == b.v.0 && a.v.1 == b.v.1 && a.v.2 == b.v.2 && a.v.3 == b.v.3 &&
           a.v.4 == b.v.4 && a.v.5 == b.v.5 && a.v.6 == b.v.6 && a.v.7 == b.v.7
}

private func cpuSegmentedScanRef(_ values: [Fr], flags: [Bool]) -> [Fr] {
    var result = [Fr](repeating: Fr.zero, count: values.count)
    var acc = Fr.zero
    for i in 0..<values.count {
        if flags[i] {
            acc = Fr.zero
        }
        acc = frAdd(acc, values[i])
        result[i] = acc
    }
    return result
}
