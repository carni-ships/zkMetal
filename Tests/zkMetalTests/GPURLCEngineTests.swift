// GPURLCEngineTests — Tests for GPU-accelerated random linear combination engine
//
// Validates correctness of RLC across CPU and GPU paths,
// including alpha power generation, batch commit combine, and edge cases.

import Foundation
import Metal
import zkMetal

public func runGPURLCEngineTests() {
    suite("GPURLCEngine")

    guard let _ = MTLCreateSystemDefaultDevice() else {
        print("  [SKIP] No Metal device available")
        return
    }

    guard let engine = try? GPURLCEngine() else {
        print("  [SKIP] Failed to create GPURLCEngine")
        return
    }

    // Helper: CPU reference RLC
    func cpuCombine(_ vectors: [[Fr]], _ powers: [Fr]) -> [Fr] {
        let k = vectors.count
        let n = vectors[0].count
        var result = [Fr](repeating: Fr.zero, count: n)
        for i in 0..<n {
            var acc = Fr.zero
            for j in 0..<k {
                acc = frAdd(acc, frMul(vectors[j][i], powers[j]))
            }
            result[i] = acc
        }
        return result
    }

    // Helper: CPU reference alpha powers
    func cpuAlphaPowers(_ alpha: Fr, _ count: Int) -> [Fr] {
        var powers = [Fr](repeating: Fr.zero, count: count)
        powers[0] = Fr.one
        for i in 1..<count {
            powers[i] = frMul(powers[i - 1], alpha)
        }
        return powers
    }

    // ================================================================
    // MARK: - Alpha powers
    // ================================================================
    suite("GPURLCEngine — Alpha powers")

    do {
        let alpha = frFromInt(7)

        // Small count (CPU path)
        let powers4 = engine.alphaPowers(alpha: alpha, count: 4)
        expectEqual(powers4.count, 4, "alpha powers count")
        expectEqual(powers4[0], Fr.one, "alpha^0 = 1")
        expectEqual(powers4[1], alpha, "alpha^1 = alpha")
        expectEqual(powers4[2], frMul(alpha, alpha), "alpha^2")
        expectEqual(powers4[3], frMul(frMul(alpha, alpha), alpha), "alpha^3")

        // Edge cases
        let powers0 = engine.alphaPowers(alpha: alpha, count: 0)
        expect(powers0.isEmpty, "alpha powers count=0")

        let powers1 = engine.alphaPowers(alpha: alpha, count: 1)
        expectEqual(powers1.count, 1, "alpha powers count=1")
        expectEqual(powers1[0], Fr.one, "alpha^0 = 1")

        // Larger count — triggers GPU path
        let powers512 = engine.alphaPowers(alpha: alpha, count: 512)
        let ref512 = cpuAlphaPowers(alpha, 512)
        expectEqual(powers512.count, 512, "alpha powers 512 count")
        var allMatch = true
        for i in 0..<512 {
            if powers512[i] != ref512[i] {
                allMatch = false
                print("  [FAIL] alpha powers mismatch at index \(i)")
                break
            }
        }
        expect(allMatch, "alpha powers 512 match CPU reference")
    }

    // ================================================================
    // MARK: - Single vector combine (trivial case)
    // ================================================================
    suite("GPURLCEngine — Single vector")

    do {
        let n = 64
        let vec = (0..<n).map { frFromInt(UInt64($0) + 1) }
        let power = frFromInt(5)

        let saved = engine.cpuThreshold
        engine.cpuThreshold = 0  // Force GPU path
        let result = engine.combine(vectors: [vec], powers: [power])
        engine.cpuThreshold = saved

        expectEqual(result.count, n, "single vector result count")
        for i in 0..<n {
            expectEqual(result[i], frMul(vec[i], power), "single vec element \(i)")
        }
    }

    // ================================================================
    // MARK: - Two-vector combine (CPU path)
    // ================================================================
    suite("GPURLCEngine — Two vectors CPU")

    do {
        let n = 32
        let v0 = (0..<n).map { frFromInt(UInt64($0) + 1) }
        let v1 = (0..<n).map { frFromInt(UInt64($0) + 100) }
        let powers = [frFromInt(3), frFromInt(7)]

        let saved = engine.cpuThreshold
        engine.cpuThreshold = 999999  // Force CPU path
        let result = engine.combine(vectors: [v0, v1], powers: powers)
        engine.cpuThreshold = saved

        let expected = cpuCombine([v0, v1], powers)
        expectEqual(result.count, n, "two vec CPU count")
        var allMatch = true
        for i in 0..<n {
            if result[i] != expected[i] {
                allMatch = false
                print("  [FAIL] two vec CPU mismatch at index \(i)")
                break
            }
        }
        expect(allMatch, "two vector CPU combine matches reference")
    }

    // ================================================================
    // MARK: - Multi-vector combine (GPU path)
    // ================================================================
    suite("GPURLCEngine — Multi-vector GPU")

    do {
        let n = 4096
        let k = 8
        var vectors = [[Fr]]()
        for j in 0..<k {
            vectors.append((0..<n).map { frFromInt(UInt64(j * n + $0) + 1) })
        }
        let alpha = frFromInt(13)
        let powers = cpuAlphaPowers(alpha, k)

        let saved = engine.cpuThreshold
        engine.cpuThreshold = 0  // Force GPU path
        let result = engine.combine(vectors: vectors, powers: powers)
        engine.cpuThreshold = saved

        let expected = cpuCombine(vectors, powers)
        expectEqual(result.count, n, "multi-vec GPU count")
        var allMatch = true
        for i in 0..<n {
            if result[i] != expected[i] {
                allMatch = false
                print("  [FAIL] multi-vec GPU mismatch at index \(i)")
                break
            }
        }
        expect(allMatch, "multi-vector GPU combine matches reference")
    }

    // ================================================================
    // MARK: - batchCommitCombine
    // ================================================================
    suite("GPURLCEngine — Batch commit combine")

    do {
        let n = 2048
        let k = 4
        var polys = [[Fr]]()
        for j in 0..<k {
            polys.append((0..<n).map { frFromInt(UInt64(j * n + $0) + 1) })
        }
        let alpha = frFromInt(11)

        let saved = engine.cpuThreshold
        engine.cpuThreshold = 0  // Force GPU
        let result = engine.batchCommitCombine(polys: polys, alpha: alpha)
        engine.cpuThreshold = saved

        // Manual reference: combine with [1, alpha, alpha^2, alpha^3]
        let powers = cpuAlphaPowers(alpha, k)
        let expected = cpuCombine(polys, powers)
        expectEqual(result.count, n, "batch commit count")
        var allMatch = true
        for i in 0..<n {
            if result[i] != expected[i] {
                allMatch = false
                print("  [FAIL] batch commit mismatch at index \(i)")
                break
            }
        }
        expect(allMatch, "batchCommitCombine matches reference")
    }

    // ================================================================
    // MARK: - Empty and edge cases
    // ================================================================
    suite("GPURLCEngine — Edge cases")

    do {
        // Empty vector
        let result = engine.combine(vectors: [[Fr]()], powers: [Fr.one])
        expect(result.isEmpty, "empty vector result is empty")

        // Single element
        let r1 = engine.combine(vectors: [[frFromInt(42)]], powers: [frFromInt(3)])
        expectEqual(r1.count, 1, "single element count")
        expectEqual(r1[0], frMul(frFromInt(42), frFromInt(3)), "single element value")

        // All-zero vectors
        let n = 128
        let zeros = [Fr](repeating: Fr.zero, count: n)
        let r2 = engine.combine(vectors: [zeros, zeros], powers: [frFromInt(5), frFromInt(7)])
        var allZero = true
        for i in 0..<n {
            if r2[i] != Fr.zero { allZero = false; break }
        }
        expect(allZero, "all-zero vectors produce zero result")

        // Power = 0 zeroes out that vector
        let ones = [Fr](repeating: Fr.one, count: n)
        let r3 = engine.combine(vectors: [ones, ones], powers: [Fr.zero, frFromInt(5)])
        let expected = [Fr](repeating: frMul(Fr.one, frFromInt(5)), count: n)
        var match = true
        for i in 0..<n {
            if r3[i] != expected[i] { match = false; break }
        }
        expect(match, "zero power zeroes out vector contribution")
    }

    // ================================================================
    // MARK: - Large-k combine (many vectors)
    // ================================================================
    suite("GPURLCEngine — Large k combine")

    do {
        let n = 1024
        let k = 32
        var vectors = [[Fr]]()
        for j in 0..<k {
            vectors.append((0..<n).map { frFromInt(UInt64(j * 1000 + $0) + 1) })
        }
        let alpha = frFromInt(3)
        let powers = cpuAlphaPowers(alpha, k)

        let saved = engine.cpuThreshold
        engine.cpuThreshold = 0
        let result = engine.combine(vectors: vectors, powers: powers)
        engine.cpuThreshold = saved

        let expected = cpuCombine(vectors, powers)
        var allMatch = true
        for i in 0..<n {
            if result[i] != expected[i] {
                allMatch = false
                print("  [FAIL] large-k mismatch at index \(i)")
                break
            }
        }
        expect(allMatch, "large-k (k=32) GPU combine matches reference")
    }

    // ================================================================
    // MARK: - Performance benchmark
    // ================================================================
    suite("GPURLCEngine — Performance")

    do {
        let n = 65536
        let k = 8
        var vectors = [[Fr]]()
        for j in 0..<k {
            vectors.append((0..<n).map { frFromInt(UInt64($0) &+ UInt64(j) &* 65536 &+ 1) })
        }
        let alpha = frFromInt(17)
        let powers = cpuAlphaPowers(alpha, k)

        // Warm up
        let _ = engine.combine(vectors: vectors, powers: powers)

        // Benchmark GPU
        let saved = engine.cpuThreshold
        engine.cpuThreshold = 0
        let t0 = CFAbsoluteTimeGetCurrent()
        let iterations = 5
        for _ in 0..<iterations {
            let _ = engine.combine(vectors: vectors, powers: powers)
        }
        let gpuTime = (CFAbsoluteTimeGetCurrent() - t0) / Double(iterations)
        engine.cpuThreshold = saved

        // Benchmark CPU
        engine.cpuThreshold = 999999
        let t1 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iterations {
            let _ = engine.combine(vectors: vectors, powers: powers)
        }
        let cpuTime = (CFAbsoluteTimeGetCurrent() - t1) / Double(iterations)
        engine.cpuThreshold = saved

        print(String(format: "  RLC n=%d k=%d: GPU %.2fms, CPU %.2fms, speedup %.1fx",
                      n, k, gpuTime * 1000, cpuTime * 1000, cpuTime / gpuTime))
        expect(true, "benchmark completed")
    }
}
