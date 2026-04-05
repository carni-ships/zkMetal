// PrefixScanEngine tests

import Foundation
import zkMetal

func runPrefixScanTests() {
    suite("PrefixScan")

    guard let engine = try? PrefixScanEngine() else {
        print("  [SKIP] No GPU available")
        return
    }

    // --- Inclusive scan uint32 (small, CPU path) ---
    do {
        let input: [UInt32] = [1, 2, 3, 4, 5]
        let result = engine.inclusiveScan(input)
        expectEqual(result.count, 5, "inclusive scan count")
        expectEqual(result[0], 1, "inclusive[0]")
        expectEqual(result[1], 3, "inclusive[1]")
        expectEqual(result[2], 6, "inclusive[2]")
        expectEqual(result[3], 10, "inclusive[3]")
        expectEqual(result[4], 15, "inclusive[4]")
        print("  [OK] inclusive scan uint32 (small)")
    }

    // --- Exclusive scan uint32 (small, CPU path) ---
    do {
        let input: [UInt32] = [1, 2, 3, 4, 5]
        let result = engine.exclusiveScan(input)
        expectEqual(result.count, 5, "exclusive scan count")
        expectEqual(result[0], 0, "exclusive[0]")
        expectEqual(result[1], 1, "exclusive[1]")
        expectEqual(result[2], 3, "exclusive[2]")
        expectEqual(result[3], 6, "exclusive[3]")
        expectEqual(result[4], 10, "exclusive[4]")
        print("  [OK] exclusive scan uint32 (small)")
    }

    // --- Inclusive scan uint32 (GPU path) ---
    do {
        let n = 4096
        var input = [UInt32](repeating: 0, count: n)
        for i in 0..<n { input[i] = UInt32(i + 1) }

        // Force GPU path
        engine.cpuThreshold = 0
        let result = engine.inclusiveScan(input)
        engine.cpuThreshold = 2048

        // Check a few values
        expectEqual(result.count, n, "GPU inclusive scan count")
        expectEqual(result[0], 1, "GPU inclusive[0]")
        expectEqual(result[1], 3, "GPU inclusive[1]")

        // Sum 1..n = n*(n+1)/2
        let expectedLast = UInt32(n * (n + 1) / 2)
        expectEqual(result[n - 1], expectedLast, "GPU inclusive[last]")
        print("  [OK] inclusive scan uint32 (GPU, n=\(n))")
    }

    // --- Exclusive scan uint32 (GPU path) ---
    do {
        let n = 4096
        var input = [UInt32](repeating: 0, count: n)
        for i in 0..<n { input[i] = UInt32(i + 1) }

        engine.cpuThreshold = 0
        let result = engine.exclusiveScan(input)
        engine.cpuThreshold = 2048

        expectEqual(result.count, n, "GPU exclusive scan count")
        expectEqual(result[0], 0, "GPU exclusive[0]")
        expectEqual(result[1], 1, "GPU exclusive[1]")
        expectEqual(result[2], 3, "GPU exclusive[2]")

        // exclusive[n-1] = sum 1..(n-1) = (n-1)*n/2
        let expectedLast = UInt32((n - 1) * n / 2)
        expectEqual(result[n - 1], expectedLast, "GPU exclusive[last]")
        print("  [OK] exclusive scan uint32 (GPU, n=\(n))")
    }

    // --- Multi-block scan (larger than one threadgroup) ---
    do {
        let n = 8192  // > 1024 elements = more than one block
        var input = [UInt32](repeating: 1, count: n)

        engine.cpuThreshold = 0
        let result = engine.inclusiveScan(input)
        engine.cpuThreshold = 2048

        expectEqual(result.count, n, "multi-block scan count")
        // All ones => inclusive scan should be [1, 2, 3, ..., n]
        var ok = true
        for i in 0..<n {
            if result[i] != UInt32(i + 1) { ok = false; break }
        }
        expect(ok, "multi-block inclusive scan all-ones")
        print("  [OK] multi-block inclusive scan uint32 (n=\(n))")
    }

    // --- Inclusive scan uint64 ---
    do {
        let input: [UInt64] = [100, 200, 300, 400, 500]
        let result = engine.inclusiveScanU64(input)
        expectEqual(result.count, 5, "inclusive u64 count")
        expectEqual(result[0], 100, "inclusive u64[0]")
        expectEqual(result[4], 1500, "inclusive u64[4]")
        print("  [OK] inclusive scan uint64 (small)")
    }

    // --- Exclusive scan uint64 ---
    do {
        let input: [UInt64] = [100, 200, 300, 400, 500]
        let result = engine.exclusiveScanU64(input)
        expectEqual(result[0], 0, "exclusive u64[0]")
        expectEqual(result[4], 1000, "exclusive u64[4]")
        print("  [OK] exclusive scan uint64 (small)")
    }

    // --- Prefix product BabyBear (small, CPU path) ---
    do {
        // BabyBear p = 2013265921
        // Product of [2, 3, 4] = [2, 6, 24] mod p
        let input: [UInt32] = [2, 3, 4, 5]
        let result = engine.prefixProductBabyBear(input)
        expectEqual(result.count, 4, "babybear product count")
        expectEqual(result[0], 2, "bb product[0]")
        expectEqual(result[1], 6, "bb product[1]")
        expectEqual(result[2], 24, "bb product[2]")
        expectEqual(result[3], 120, "bb product[3]")
        print("  [OK] prefix product BabyBear (small)")
    }

    // --- Empty input ---
    do {
        let empty: [UInt32] = []
        expectEqual(engine.inclusiveScan(empty).count, 0, "empty inclusive")
        expectEqual(engine.exclusiveScan(empty).count, 0, "empty exclusive")
        expectEqual(engine.inclusiveScanU64([]).count, 0, "empty u64 inclusive")
        expectEqual(engine.prefixProductBabyBear([]).count, 0, "empty bb product")
        print("  [OK] empty inputs")
    }

    // --- Single element ---
    do {
        expectEqual(engine.inclusiveScan([42]), [42], "single inclusive")
        expectEqual(engine.exclusiveScan([42]), [0], "single exclusive")
        print("  [OK] single element")
    }

    // --- Consistency: GPU vs CPU for inclusive scan ---
    do {
        let n = 4096
        var input = [UInt32](repeating: 0, count: n)
        for i in 0..<n { input[i] = UInt32.random(in: 0..<1000) }

        // CPU path
        engine.cpuThreshold = n + 1
        let cpuResult = engine.inclusiveScan(input)

        // GPU path
        engine.cpuThreshold = 0
        let gpuResult = engine.inclusiveScan(input)
        engine.cpuThreshold = 2048

        var match = true
        for i in 0..<n {
            if cpuResult[i] != gpuResult[i] { match = false; break }
        }
        expect(match, "GPU vs CPU inclusive scan consistency")
        print("  [OK] GPU vs CPU consistency (n=\(n))")
    }

    // --- Performance: large scan ---
    do {
        let n = 1 << 20  // 1M elements
        let input = [UInt32](repeating: 1, count: n)

        engine.cpuThreshold = 0
        let t0 = CFAbsoluteTimeGetCurrent()
        let result = engine.inclusiveScan(input)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        engine.cpuThreshold = 2048

        expectEqual(result[n - 1], UInt32(n), "1M scan last element")
        print(String(format: "  [OK] inclusive scan 1M elements: %.2fms", elapsed * 1000))
    }
}
