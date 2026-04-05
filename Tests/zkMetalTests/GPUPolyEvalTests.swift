import zkMetal
import Foundation

// MARK: - Test helpers

/// CPU Horner evaluation for BabyBear (reference implementation)
private func cpuHornerBb(coeffs: [UInt32], x: UInt32) -> UInt32 {
    let p = UInt64(Bb.P)
    var result = UInt64(coeffs.last!)
    for i in stride(from: coeffs.count - 2, through: 0, by: -1) {
        result = (result * UInt64(x) + UInt64(coeffs[i])) % p
    }
    return UInt32(result)
}

/// CPU Horner evaluation for Goldilocks (reference)
private func cpuHornerGl(coeffs: [UInt64], x: UInt64) -> UInt64 {
    var result = Gl(v: coeffs.last!)
    for i in stride(from: coeffs.count - 2, through: 0, by: -1) {
        result = glAdd(glMul(result, Gl(v: x)), Gl(v: coeffs[i]))
    }
    return result.v
}

/// CPU Horner evaluation for BN254 Fr (reference)
private func cpuHornerFr(coeffs: [Fr], x: Fr) -> Fr {
    var result = coeffs.last!
    for i in stride(from: coeffs.count - 2, through: 0, by: -1) {
        result = frAdd(frMul(result, x), coeffs[i])
    }
    return result
}

/// Simple LCG for test reproducibility
private struct TestRNG {
    var state: UInt64

    mutating func next32() -> UInt32 {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return UInt32(state >> 33)
    }

    mutating func next64() -> UInt64 {
        let hi = UInt64(next32()) << 32
        return hi | UInt64(next32())
    }

    mutating func nextBb() -> UInt32 {
        return next32() % Bb.P
    }

    mutating func nextGl() -> UInt64 {
        let v = next64()
        return v >= Gl.P ? v - Gl.P : v
    }

    mutating func nextFr() -> Fr {
        // Generate a random Fr in Montgomery form by multiplying random value by R^2
        let raw = Fr(v: (next32() & 0x0FFFFFFF, next32(), next32(), next32(),
                         next32(), next32(), next32(), next32() & 0x0FFFFFFF))
        return frMul(raw, Fr.from64(Fr.R2_MOD_R))
    }
}

// MARK: - Public test entry point

public func runGPUPolyEvalTests() {
    testSingleEvalBabyBear()
    testSingleEvalGoldilocks()
    testSingleEvalBN254()
    testBatchEvalBabyBear()
    testBatchEvalGoldilocks()
    testBatchEvalBN254()
    testBatchConsistency()
    testRootsOfUnityBabyBear()
    testLargePolyPerformance()
}

// MARK: - Single polynomial evaluation tests

private func testSingleEvalBabyBear() {
    suite("GPUPolyEval BabyBear single")
    do {
        let engine = try GPUPolyEvalEngine()
        var rng = TestRNG(state: 0xCAFEBABE)

        // Small polynomial: p(x) = 1 + 2x + 3x^2 + 4x^3
        let coeffs: [UInt32] = [1, 2, 3, 4]
        let points: [UInt32] = [0, 1, 2, 3, 5, 10, 100]

        // Force GPU path by using >= 64 points (pad with more)
        var allPoints = points
        for _ in 0..<(64 - points.count) { allPoints.append(rng.nextBb()) }

        let results = try engine.evaluate(coeffs: coeffs, points: allPoints, field: .babybear)
        expectEqual(results.count, allPoints.count, "BabyBear single eval result count")

        // Verify first few against CPU reference
        for i in 0..<points.count {
            let expected = cpuHornerBb(coeffs: coeffs, x: points[i])
            expectEqual(results[i], expected, "BabyBear eval at x=\(points[i])")
        }

        // Random polynomial degree 128
        let deg = 128
        var rcoeffs = [UInt32]()
        for _ in 0..<deg { rcoeffs.append(rng.nextBb()) }
        var rpoints = [UInt32]()
        for _ in 0..<128 { rpoints.append(rng.nextBb()) }

        let rresults = try engine.evaluate(coeffs: rcoeffs, points: rpoints, field: .babybear)
        expectEqual(rresults.count, 128, "BabyBear random eval count")

        // Spot check a few
        for i in 0..<4 {
            let expected = cpuHornerBb(coeffs: rcoeffs, x: rpoints[i])
            expectEqual(rresults[i], expected, "BabyBear random eval[\(i)]")
        }

    } catch { expect(false, "BabyBear single eval error: \(error)") }
}

private func testSingleEvalGoldilocks() {
    suite("GPUPolyEval Goldilocks single")
    do {
        let engine = try GPUPolyEvalEngine()
        var rng = TestRNG(state: 0xDEADBEEF)

        // p(x) = 1 + 2x + 3x^2
        let coeffsU64: [UInt64] = [1, 2, 3]
        // Convert to UInt32 pairs (little-endian)
        var coeffs = [UInt32]()
        for c in coeffsU64 {
            coeffs.append(UInt32(c & 0xFFFFFFFF))
            coeffs.append(UInt32(c >> 32))
        }

        var pointsU64 = [UInt64]()
        var pointsU32 = [UInt32]()
        for _ in 0..<64 {
            let p = rng.nextGl()
            pointsU64.append(p)
            pointsU32.append(UInt32(p & 0xFFFFFFFF))
            pointsU32.append(UInt32(p >> 32))
        }

        let results = try engine.evaluate(coeffs: coeffs, points: pointsU32, field: .goldilocks)
        expectEqual(results.count, 128, "Goldilocks single eval result count (64 points x 2 words)")

        for i in 0..<4 {
            let expected = cpuHornerGl(coeffs: coeffsU64, x: pointsU64[i])
            let got = UInt64(results[i * 2]) | (UInt64(results[i * 2 + 1]) << 32)
            expectEqual(got, expected, "Goldilocks eval[\(i)]")
        }

    } catch { expect(false, "Goldilocks single eval error: \(error)") }
}

private func testSingleEvalBN254() {
    suite("GPUPolyEval BN254 single")
    do {
        let engine = try GPUPolyEvalEngine()
        var rng = TestRNG(state: 0x12345678)

        // p(x) = a0 + a1*x + a2*x^2, small coefficients
        let a0 = frFromInt(1)
        let a1 = frFromInt(2)
        let a2 = frFromInt(3)
        let coeffsFr = [a0, a1, a2]

        // Convert to UInt32 array
        func frToU32(_ f: Fr) -> [UInt32] {
            [f.v.0, f.v.1, f.v.2, f.v.3, f.v.4, f.v.5, f.v.6, f.v.7]
        }
        var coeffs = [UInt32]()
        for c in coeffsFr { coeffs.append(contentsOf: frToU32(c)) }

        // Generate 64 random points
        var pointsFr = [Fr]()
        var points = [UInt32]()
        for _ in 0..<64 {
            let p = rng.nextFr()
            pointsFr.append(p)
            points.append(contentsOf: frToU32(p))
        }

        let results = try engine.evaluate(coeffs: coeffs, points: points, field: .bn254)
        expectEqual(results.count, 64 * 8, "BN254 single eval result count")

        // Verify against CPU reference
        for i in 0..<4 {
            let expected = cpuHornerFr(coeffs: coeffsFr, x: pointsFr[i])
            let base = i * 8
            let got = Fr(v: (results[base], results[base+1], results[base+2], results[base+3],
                             results[base+4], results[base+5], results[base+6], results[base+7]))
            expectEqual(got, expected, "BN254 eval[\(i)]")
        }

    } catch { expect(false, "BN254 single eval error: \(error)") }
}

// MARK: - Batch evaluation tests

private func testBatchEvalBabyBear() {
    suite("GPUPolyEval BabyBear batch")
    do {
        let engine = try GPUPolyEvalEngine()
        var rng = TestRNG(state: 0xABCD1234)

        let numPolys = 4
        let degree = 64
        let numPoints = 128

        var polys = [[UInt32]]()
        for _ in 0..<numPolys {
            var coeffs = [UInt32]()
            for _ in 0..<degree { coeffs.append(rng.nextBb()) }
            polys.append(coeffs)
        }

        var points = [UInt32]()
        for _ in 0..<numPoints { points.append(rng.nextBb()) }

        let results = try engine.evaluateBatch(polys: polys, points: points, field: .babybear)
        expectEqual(results.count, numPolys, "BabyBear batch result poly count")

        for m in 0..<numPolys {
            expectEqual(results[m].count, numPoints, "BabyBear batch poly[\(m)] result count")
            // Verify first 4 points
            for p in 0..<4 {
                let expected = cpuHornerBb(coeffs: polys[m], x: points[p])
                expectEqual(results[m][p], expected, "BabyBear batch poly[\(m)] point[\(p)]")
            }
        }

    } catch { expect(false, "BabyBear batch eval error: \(error)") }
}

private func testBatchEvalGoldilocks() {
    suite("GPUPolyEval Goldilocks batch")
    do {
        let engine = try GPUPolyEvalEngine()
        var rng = TestRNG(state: 0xFEED9999)

        let numPolys = 3
        let degree = 32
        let numPoints = 128

        var polysU64 = [[UInt64]]()
        var polysU32 = [[UInt32]]()
        for _ in 0..<numPolys {
            var u64 = [UInt64]()
            var u32 = [UInt32]()
            for _ in 0..<degree {
                let c = rng.nextGl()
                u64.append(c)
                u32.append(UInt32(c & 0xFFFFFFFF))
                u32.append(UInt32(c >> 32))
            }
            polysU64.append(u64)
            polysU32.append(u32)
        }

        var pointsU64 = [UInt64]()
        var pointsU32 = [UInt32]()
        for _ in 0..<numPoints {
            let p = rng.nextGl()
            pointsU64.append(p)
            pointsU32.append(UInt32(p & 0xFFFFFFFF))
            pointsU32.append(UInt32(p >> 32))
        }

        let results = try engine.evaluateBatch(polys: polysU32, points: pointsU32, field: .goldilocks)
        expectEqual(results.count, numPolys, "Goldilocks batch result poly count")

        for m in 0..<numPolys {
            expectEqual(results[m].count, numPoints * 2, "Goldilocks batch poly[\(m)] result count")
            for p in 0..<4 {
                let expected = cpuHornerGl(coeffs: polysU64[m], x: pointsU64[p])
                let got = UInt64(results[m][p * 2]) | (UInt64(results[m][p * 2 + 1]) << 32)
                expectEqual(got, expected, "Goldilocks batch poly[\(m)] point[\(p)]")
            }
        }

    } catch { expect(false, "Goldilocks batch eval error: \(error)") }
}

private func testBatchEvalBN254() {
    suite("GPUPolyEval BN254 batch")
    do {
        let engine = try GPUPolyEvalEngine()
        var rng = TestRNG(state: 0x77778888)

        let numPolys = 3
        let degree = 16
        let numPoints = 64

        func frToU32(_ f: Fr) -> [UInt32] {
            [f.v.0, f.v.1, f.v.2, f.v.3, f.v.4, f.v.5, f.v.6, f.v.7]
        }

        var polysFr = [[Fr]]()
        var polysU32 = [[UInt32]]()
        for _ in 0..<numPolys {
            var fr = [Fr]()
            var u32 = [UInt32]()
            for _ in 0..<degree {
                let c = rng.nextFr()
                fr.append(c)
                u32.append(contentsOf: frToU32(c))
            }
            polysFr.append(fr)
            polysU32.append(u32)
        }

        var pointsFr = [Fr]()
        var pointsU32 = [UInt32]()
        for _ in 0..<numPoints {
            let p = rng.nextFr()
            pointsFr.append(p)
            pointsU32.append(contentsOf: frToU32(p))
        }

        let results = try engine.evaluateBatch(polys: polysU32, points: pointsU32, field: .bn254)
        expectEqual(results.count, numPolys, "BN254 batch result poly count")

        for m in 0..<numPolys {
            expectEqual(results[m].count, numPoints * 8, "BN254 batch poly[\(m)] result count")
            for p in 0..<4 {
                let expected = cpuHornerFr(coeffs: polysFr[m], x: pointsFr[p])
                let base = p * 8
                let got = Fr(v: (results[m][base], results[m][base+1], results[m][base+2], results[m][base+3],
                                 results[m][base+4], results[m][base+5], results[m][base+6], results[m][base+7]))
                expectEqual(got, expected, "BN254 batch poly[\(m)] point[\(p)]")
            }
        }

    } catch { expect(false, "BN254 batch eval error: \(error)") }
}

// MARK: - Batch consistency: batch of 1 == single eval

private func testBatchConsistency() {
    suite("GPUPolyEval batch-single consistency")
    do {
        let engine = try GPUPolyEvalEngine()
        var rng = TestRNG(state: 0xAAAABBBB)

        let degree = 64
        let numPoints = 128

        var coeffs = [UInt32]()
        for _ in 0..<degree { coeffs.append(rng.nextBb()) }

        var points = [UInt32]()
        for _ in 0..<numPoints { points.append(rng.nextBb()) }

        let single = try engine.evaluate(coeffs: coeffs, points: points, field: .babybear)
        let batch = try engine.evaluateBatch(polys: [coeffs], points: points, field: .babybear)

        expectEqual(batch.count, 1, "Batch of 1 returns 1 result array")
        expectEqual(batch[0].count, single.count, "Batch of 1 same length as single")

        var match = true
        for i in 0..<single.count {
            if single[i] != batch[0][i] { match = false; break }
        }
        expect(match, "Batch of 1 matches single eval (BabyBear)")

    } catch { expect(false, "Batch consistency error: \(error)") }
}

// MARK: - Roots of unity test (should match NTT output)

private func testRootsOfUnityBabyBear() {
    suite("GPUPolyEval BabyBear roots of unity")
    do {
        let engine = try GPUPolyEvalEngine()

        // Polynomial of degree 8: evaluate at 8th roots of unity
        // This should match NTT output
        let logN = 3
        let n = 1 << logN

        let coeffs: [UInt32] = [1, 2, 3, 4, 5, 6, 7, 8]

        // Compute 8th root of unity: omega = ROOT_OF_UNITY^(2^(TWO_ADICITY - logN))
        var omega = Bb(v: Bb.ROOT_OF_UNITY)
        for _ in 0..<(Bb.TWO_ADICITY - logN) {
            omega = bbSqr(omega)
        }

        // Generate roots: omega^0, omega^1, ..., omega^7
        var points = [UInt32]()
        var omPow = Bb.one
        for _ in 0..<n {
            points.append(omPow.v)
            omPow = bbMul(omPow, omega)
        }

        // Evaluate polynomial at all roots using engine (will use CPU fallback since < 64)
        // But we can verify correctness of the CPU path too
        let results = try engine.evaluate(coeffs: coeffs, points: points, field: .babybear)
        expectEqual(results.count, n, "Roots of unity eval count")

        // Verify against direct Horner
        for i in 0..<n {
            let expected = cpuHornerBb(coeffs: coeffs, x: points[i])
            expectEqual(results[i], expected, "Root of unity eval[\(i)]")
        }

        // Verify omega^n == 1
        expectEqual(omPow.v, Bb.one.v, "omega^n should be 1")

    } catch { expect(false, "Roots of unity error: \(error)") }
}

// MARK: - Large polynomial performance test

private func testLargePolyPerformance() {
    suite("GPUPolyEval large poly performance")
    do {
        let engine = try GPUPolyEvalEngine()
        var rng = TestRNG(state: 0x99887766)

        // Degree 2^16, 2^10 points
        let degree = 1 << 16
        let numPoints = 1 << 10

        var coeffs = [UInt32]()
        coeffs.reserveCapacity(degree)
        for _ in 0..<degree { coeffs.append(rng.nextBb()) }

        var points = [UInt32]()
        points.reserveCapacity(numPoints)
        for _ in 0..<numPoints { points.append(rng.nextBb()) }

        // Warm up
        _ = try engine.evaluate(coeffs: coeffs, points: points, field: .babybear)

        // Timed run
        let t0 = CFAbsoluteTimeGetCurrent()
        let results = try engine.evaluate(coeffs: coeffs, points: points, field: .babybear)
        let elapsed = CFAbsoluteTimeGetCurrent() - t0
        expectEqual(results.count, numPoints, "Large poly eval count")
        print(String(format: "  BabyBear deg=65536, pts=1024: %.1fms", elapsed * 1000))

        // Spot check first result
        let expected = cpuHornerBb(coeffs: coeffs, x: points[0])
        expectEqual(results[0], expected, "Large poly eval[0] correctness")

    } catch { expect(false, "Large poly performance error: \(error)") }
}
